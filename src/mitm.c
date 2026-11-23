/*
 * Distributed meet-in-the-middle search for a "golden collision".
 *
 * Given f, g : {0,1}^n -> {0,1}^n and a predicate pi, find (x, y) with
 * f(x) = g(y) and pi(x, y). Here f encrypts a fixed plaintext under key x,
 * g decrypts a fixed ciphertext under key y (Speck64/128), and pi checks a
 * second plaintext/ciphertext pair.
 *
 * f and g are truncations to n bits of a 64-bit middle state. The search
 * keeps the full state w: its low n bits z decide where a pair goes, and
 * 63 - n of the discarded bits are stored as a fingerprint. A candidate
 * pair then has to match on 63 bits of the middle state, not only on n, so
 * almost no wrong candidate reaches the final check, which would otherwise
 * cost two more encryptions per key.
 *
 * The dictionary z = f(x) -> x is sharded across MPI ranks by z mod p. The
 * key space is processed in blocks: each rank computes a block of f(x) with
 * OpenMP, sends every pair to the rank that owns z, and the owner inserts
 * it. The search side does the same with g(y) and probes on arrival.
 *
 * With --passes K the dictionary is split into K classes of z. Pass j keeps
 * only the pairs whose z falls in class j, so each rank holds 1/K of its
 * shard at a time. f and g are recomputed in every pass: memory drops by K,
 * computation grows by K, and the volume of data exchanged is unchanged.
 */

#include <err.h>
#include <getopt.h>
#include <inttypes.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"
#include "exchange.h"
#include "speck.h"

#define MAX_RESULTS 16
#define MAX_CANDIDATES 256
#define PREFETCH_AHEAD 16   /* pairs ahead whose slot is requested from memory early */

static int n = 0;
static uint64_t mask;
static uint32_t C[2][2];

/******************************* the problem *********************************/

static inline void expand_key(uint64_t k, uint32_t rk[27])
{
	uint32_t K[4] = {k & 0xffffffff, k >> 32, 0, 0};
	speck_key_schedule(K, rk);
}

/* full middle state for f: encryption of P0 under k; f(k) is its low n bits */
static inline uint64_t f_state(uint64_t k)
{
	uint32_t rk[27], ct[2];
	expand_key(k, rk);
	speck_encrypt(SPECK_P0, ct, rk);
	return (uint64_t)ct[0] | ((uint64_t)ct[1] << 32);
}

/* full middle state for g: decryption of C0 under k; g(k) is its low n bits */
static inline uint64_t g_state(uint64_t k)
{
	uint32_t rk[27], pt[2];
	expand_key(k, rk);
	speck_decrypt(pt, C[0], rk);
	return (uint64_t)pt[0] | ((uint64_t)pt[1] << 32);
}

/* pi(k1, k2): the second pair also matches */
static bool is_good_pair(uint64_t k1, uint64_t k2)
{
	uint32_t rka[27], rkb[27], mid[2], ct[2];
	expand_key(k1, rka);
	expand_key(k2, rkb);
	speck_encrypt(SPECK_P1, mid, rka);
	speck_encrypt(mid, ct, rkb);
	return ct[0] == C[1][0] && ct[1] == C[1][1];
}

/******************************** the search *********************************/

typedef struct {
	int rank, p;
	int passes;
	uint64_t block;        /* keys per block and per rank */
	uint64_t start, end;   /* this rank's slice of the key space */
	uint64_t rounds;       /* blocks per pass, identical on every rank */
	uint64_t *w;           /* middle states of the current block */
	uint64_t *send;        /* pairs grouped by destination */
	exchange_t ex;
	dict_t dict;
	double t_eval, t_comm, t_insert, t_probe;
	uint64_t candidates;
	int nres;
	uint64_t res1[MAX_RESULTS], res2[MAX_RESULTS];
} search_t;

static inline int pass_of(uint64_t w, int passes)
{
	return passes == 1 ? 0 : (int)((mix64(w & mask) >> 40) % (uint64_t)passes);
}

static inline int owner(uint64_t w, int p)
{
	return (int)((w & mask) % (uint64_t)p);
}

/*
 * Evaluate fun on block `round` of this rank's slice, keep the states whose
 * z is in class `pass`, and group the (state, key) pairs by owner rank.
 */
static void prepare_block(search_t *s, uint64_t (*fun)(uint64_t), uint64_t round, int pass)
{
	uint64_t base = s->start + round * s->block;
	uint64_t count = base >= s->end ? 0 : (s->end - base < s->block ? s->end - base : s->block);

	double t0 = MPI_Wtime();
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < count; i++)
		s->w[i] = fun(base + i);
	s->t_eval += MPI_Wtime() - t0;

	int p = s->p;
	memset(s->ex.send_count, 0, p * sizeof(int));
	for (uint64_t i = 0; i < count; i++)
		if (pass_of(s->w[i], s->passes) == pass)
			s->ex.send_count[owner(s->w[i], p)] += 2;

	int offset = 0;
	for (int r = 0; r < p; r++) {
		s->ex.send_displ[r] = offset;
		offset += s->ex.send_count[r];
	}

	int *fill = malloc(p * sizeof(int));
	memcpy(fill, s->ex.send_displ, p * sizeof(int));
	for (uint64_t i = 0; i < count; i++) {
		uint64_t w = s->w[i];
		if (pass_of(w, s->passes) != pass)
			continue;
		int r = owner(w, p);
		s->send[fill[r]++] = w;
		s->send[fill[r]++] = base + i;
	}
	free(fill);
}

static void build_shard(search_t *s, int pass)
{
	for (uint64_t round = 0; round < s->rounds; round++) {
		prepare_block(s, f_state, round, pass);

		double t0 = MPI_Wtime();
		size_t m = exchange(&s->ex, s->send);
		double t1 = MPI_Wtime();
		s->t_comm += t1 - t0;

		if (dict_reserve(&s->dict, m / 2) < 0)
			errx(1, "rank %d: dictionary full, use more ranks or more passes", s->rank);
		const uint64_t *recv = s->ex.recv;
#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < m; i += 2) {
			if (i + 2 * PREFETCH_AHEAD < m)
				dict_prefetch_write(&s->dict, recv[i + 2 * PREFETCH_AHEAD] & mask);
			uint64_t w = recv[i];
			dict_insert_concurrent(&s->dict, w & mask, w >> n, recv[i + 1]);
		}
		s->t_insert += MPI_Wtime() - t1;
	}
}

static void probe_shard(search_t *s, int pass)
{
	for (uint64_t round = 0; round < s->rounds; round++) {
		prepare_block(s, g_state, round, pass);

		double t0 = MPI_Wtime();
		size_t m = exchange(&s->ex, s->send);
		double t1 = MPI_Wtime();
		s->t_comm += t1 - t0;

		const uint64_t *recv = s->ex.recv;
		uint64_t cand = 0;
#pragma omp parallel for schedule(static) reduction(+:cand)
		for (size_t i = 0; i < m; i += 2) {
			if (i + 2 * PREFETCH_AHEAD < m)
				dict_prefetch(&s->dict, recv[i + 2 * PREFETCH_AHEAD] & mask);
			uint64_t w = recv[i], y = recv[i + 1];
			uint64_t x[MAX_CANDIDATES];
			int nx = dict_probe(&s->dict, w & mask, w >> n, MAX_CANDIDATES, x);
			if (nx < 0)
				errx(1, "too many candidates for one key");
			cand += (uint64_t)nx;
			for (int j = 0; j < nx; j++) {
				if (!is_good_pair(x[j], y))
					continue;
#pragma omp critical(results)
				if (s->nres < MAX_RESULTS) {
					s->res1[s->nres] = x[j];
					s->res2[s->nres] = y;
					s->nres++;
				}
			}
		}
		s->candidates += cand;
		s->t_probe += MPI_Wtime() - t1;
	}
}

/******************************* command line ********************************/

static void usage(const char *prog)
{
	printf("usage: %s --n N --C0 HEX --C1 HEX [--passes K] [--block B] [--load L]\n\n", prog);
	printf("  --n N        size of the instance in bits (at most 56)\n");
	printf("  --C0, --C1   the two ciphertexts, in hexadecimal\n");
	printf("  --passes K   split the dictionary into K passes [1]\n");
	printf("  --block B    process 2^B keys per rank and per block [20]\n");
	printf("  --load L     target load factor of the hash table [0.8]\n");
	exit(1);
}

static void parse(int argc, char **argv, int *passes, int *log_block, double *load)
{
	struct option opts[] = {
		{"n", required_argument, NULL, 'n'},
		{"C0", required_argument, NULL, '0'},
		{"C1", required_argument, NULL, '1'},
		{"passes", required_argument, NULL, 'k'},
		{"block", required_argument, NULL, 'b'},
		{"load", required_argument, NULL, 'l'},
		{NULL, 0, NULL, 0}
	};
	int ch, set = 0;
	while ((ch = getopt_long(argc, argv, "", opts, NULL)) != -1) {
		switch (ch) {
		case 'n': n = atoi(optarg); break;
		case 'k': *passes = atoi(optarg); break;
		case 'b': *log_block = atoi(optarg); break;
		case 'l': *load = atof(optarg); break;
		case '0':
		case '1': {
			uint64_t c = strtoull(optarg, NULL, 16);
			int i = ch - '0';
			C[i][0] = c & 0xffffffff;
			C[i][1] = c >> 32;
			set |= 1 << i;
			break;
		}
		default: usage(argv[0]);
		}
	}
	if (n < 1 || n > 56 || set != 3 || *passes < 1 || *log_block < 4 || *load <= 0 || *load >= 1)
		usage(argv[0]);
	mask = (1ull << n) - 1;
}

/********************************** main *************************************/

static void human_bytes(double b, char *out)
{
	const char *unit[] = {"B", "KB", "MB", "GB", "TB"};
	int u = 0;
	while (b >= 1000 && u < 4) {
		b /= 1000;
		u++;
	}
	sprintf(out, "%.1f %s", b, unit[u]);
}

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

	search_t s = {0};
	MPI_Comm_rank(MPI_COMM_WORLD, &s.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &s.p);

	int log_block = 20;
	double load = 0.8;
	s.passes = 1;
	parse(argc, argv, &s.passes, &log_block, &load);

	uint64_t N = 1ull << n;
	uint64_t per_rank = (N + s.p - 1) / s.p;
	s.block = 1ull << log_block;
	if (s.block > per_rank)
		s.block = per_rank;
	s.start = (s.rank * N) / s.p;
	s.end = ((s.rank + 1) * N) / s.p;
	s.rounds = (per_rank + s.block - 1) / s.block;

	/* expected entries per rank and per pass, at the target load factor */
	uint64_t expected = N / ((uint64_t)s.p * s.passes);
	uint64_t slots = (uint64_t)(expected / load) + 4096;
	if (dict_init(&s.dict, slots, n) < 0)
		errx(1, "rank %d: cannot allocate the dictionary", s.rank);

	s.w = malloc(s.block * sizeof(uint64_t));
	s.send = malloc(2 * s.block * sizeof(uint64_t));
	exchange_init(&s.ex, s.p);

	if (s.rank == 0) {
		char mem[32];
		human_bytes((double)slots * sizeof(uint64_t), mem);
		printf("n = %d, %d ranks x %d threads, %d pass(es), blocks of %" PRIu64 " keys\n",
		       n, s.p, omp_get_max_threads(), s.passes, s.block);
		printf("dictionary per rank: %s\n", mem);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t_start = MPI_Wtime();

	for (int pass = 0; pass < s.passes; pass++) {
		if (pass > 0)
			dict_clear(&s.dict);
		build_shard(&s, pass);
		probe_shard(&s, pass);
	}

	double elapsed = MPI_Wtime() - t_start;

	/* timings: maximum over ranks */
	double t_local[5] = {s.t_eval, s.t_comm, s.t_insert, s.t_probe, elapsed};
	double t_max[5];
	MPI_Reduce(t_local, t_max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	uint64_t cand = 0;
	MPI_Reduce(&s.candidates, &cand, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

	/* results: gather on rank 0 */
	int *counts = s.rank == 0 ? malloc(s.p * sizeof(int)) : NULL;
	int *displ = s.rank == 0 ? malloc(s.p * sizeof(int)) : NULL;
	MPI_Gather(&s.nres, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int total = 0;
	if (s.rank == 0)
		for (int r = 0; r < s.p; r++) {
			displ[r] = total;
			total += counts[r];
		}
	uint64_t *k1 = s.rank == 0 ? malloc((total + 1) * sizeof(uint64_t)) : NULL;
	uint64_t *k2 = s.rank == 0 ? malloc((total + 1) * sizeof(uint64_t)) : NULL;
	MPI_Gatherv(s.res1, s.nres, MPI_UINT64_T, k1, counts, displ, MPI_UINT64_T, 0, MPI_COMM_WORLD);
	MPI_Gatherv(s.res2, s.nres, MPI_UINT64_T, k2, counts, displ, MPI_UINT64_T, 0, MPI_COMM_WORLD);

	if (s.rank == 0) {
		printf("time: %.2f s (evaluation %.2f, exchange %.2f, insertion %.2f, probing %.2f; max over ranks)\n",
		       t_max[4], t_max[0], t_max[1], t_max[2], t_max[3]);
		printf("candidates checked: %" PRIu64 "\n", cand);
		for (int i = 0; i < total; i++)
			printf("solution: %" PRIx64 " %" PRIx64 "\n", k1[i], k2[i]);
		if (total == 0)
			printf("no solution found\n");
		free(counts);
		free(displ);
		free(k1);
		free(k2);
	}

	dict_free(&s.dict);
	exchange_free(&s.ex);
	free(s.w);
	free(s.send);
	MPI_Finalize();
	return 0;
}
