/*
 * Generate a test instance with a known solution.
 *
 *   gen_instance <n> [seed]
 *
 * Draws two n-bit keys k1, k2, encrypts the two fixed plaintexts with
 * Speck64/128 under k1 then k2, and prints "C0 C1 k1 k2" in hexadecimal.
 * The search program must find (k1, k2) from (n, C0, C1).
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/speck.h"

static uint64_t splitmix64(uint64_t *s)
{
	uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31);
}

static uint64_t double_encrypt(const uint32_t pt[2], uint64_t k1, uint64_t k2)
{
	uint32_t ka[4] = {k1 & 0xffffffff, k1 >> 32, 0, 0};
	uint32_t kb[4] = {k2 & 0xffffffff, k2 >> 32, 0, 0};
	uint32_t rka[27], rkb[27], mid[2], ct[2];

	speck_key_schedule(ka, rka);
	speck_key_schedule(kb, rkb);
	speck_encrypt(pt, mid, rka);
	speck_encrypt(mid, ct, rkb);
	return (uint64_t)ct[0] | ((uint64_t)ct[1] << 32);
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: %s n [seed]\n", argv[0]);
		return 1;
	}
	int n = atoi(argv[1]);
	uint64_t seed = argc > 2 ? strtoull(argv[2], NULL, 10) : 1;
	uint64_t mask = (n == 64) ? ~0ull : (1ull << n) - 1;

	uint64_t k1 = splitmix64(&seed) & mask;
	uint64_t k2 = splitmix64(&seed) & mask;

	printf("%016" PRIx64 " %016" PRIx64 " %" PRIx64 " %" PRIx64 "\n",
	       double_encrypt(SPECK_P0, k1, k2), double_encrypt(SPECK_P1, k1, k2), k1, k2);
	return 0;
}
