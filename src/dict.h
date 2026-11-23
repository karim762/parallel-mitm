/*
 * Hash table for the local shard of the dictionary z -> x.
 *
 * One entry is a single 64-bit word: the low v = n + 1 bits hold x + 1 (so
 * that 0 marks an empty slot) and the remaining 63 - n bits hold a
 * fingerprint supplied by the caller. Slots are found by linear probing
 * from hash(z). A probe returns the x stored under the same z and the same
 * fingerprint; the caller checks them.
 *
 * Compared with a {u32 key, u64 value} entry this uses 8 bytes per slot
 * instead of 12. It requires n <= 56, which leaves at least 7 fingerprint
 * bits.
 */

#ifndef DICT_H
#define DICT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	uint64_t *slot;
	uint64_t size;      /* number of slots */
	uint64_t used;      /* number of entries */
	int vbits;          /* bits used by x + 1 */
	uint64_t vmask;
} dict_t;

static inline uint64_t mix64(uint64_t x)
{
	x ^= x >> 33;
	x *= 0xff51afd7ed558ccdull;
	x ^= x >> 33;
	x *= 0xc4ceb9fe1a85ec53ull;
	x ^= x >> 33;
	return x;
}

/* keep the first 63 - n bits of fp, in the high part of the word */
static inline uint64_t fingerprint(const dict_t *d, uint64_t fp)
{
	return fp << d->vbits;
}

/*
 * Zero the table with all threads. Besides clearing it, this makes the
 * operating system map the pages now, in parallel, instead of one page fault
 * at a time during the first insertions.
 */
static void dict_clear(dict_t *d)
{
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < d->size; i++)
		d->slot[i] = 0;
	d->used = 0;
}

static int dict_init(dict_t *d, uint64_t size, int n)
{
	d->size = size;
	d->vbits = n + 1;
	d->vmask = (1ull << d->vbits) - 1;
	d->slot = malloc(size * sizeof(uint64_t));
	if (d->slot == NULL)
		return -1;
	dict_clear(d);
	return 0;
}

static void dict_free(dict_t *d)
{
	free(d->slot);
	d->slot = NULL;
}

/* ask for the first slot of z before it is needed; lookups are memory bound */
static inline void dict_prefetch(const dict_t *d, uint64_t z)
{
	__builtin_prefetch(&d->slot[mix64(z) % d->size], 0);
}

static inline void dict_prefetch_write(dict_t *d, uint64_t z)
{
	__builtin_prefetch(&d->slot[mix64(z) % d->size], 1);
}

/* returns -1 when the table is full */
static inline int dict_insert(dict_t *d, uint64_t z, uint64_t fp, uint64_t x)
{
	if (d->used + 1 >= d->size)
		return -1;
	uint64_t h = mix64(z) % d->size;
	while (d->slot[h] != 0)
		if (++h == d->size)
			h = 0;
	d->slot[h] = fingerprint(d, fp) | (x + 1);
	d->used++;
	return 0;
}

/*
 * Insertion that several threads can run at the same time: a slot is taken
 * with a compare-and-swap, and entries are never removed, so a thread that
 * loses the race simply moves on to the next slot. The caller must reserve
 * room beforehand with dict_reserve.
 */
static inline void dict_insert_concurrent(dict_t *d, uint64_t z, uint64_t fp, uint64_t x)
{
	uint64_t entry = fingerprint(d, fp) | (x + 1);
	uint64_t h = mix64(z) % d->size;
	for (;;) {
		uint64_t expected = 0;
		if (__atomic_load_n(&d->slot[h], __ATOMIC_RELAXED) == 0 &&
		    __atomic_compare_exchange_n(&d->slot[h], &expected, entry, false,
		                                __ATOMIC_RELAXED, __ATOMIC_RELAXED))
			return;
		if (++h == d->size)
			h = 0;
	}
}

/* account for m entries about to be inserted; returns -1 if they do not fit */
static inline int dict_reserve(dict_t *d, uint64_t m)
{
	if (d->used + m >= d->size)
		return -1;
	d->used += m;
	return 0;
}

/* writes up to maxval candidates x for (z, fp); returns their number, or -1 */
static inline int dict_probe(const dict_t *d, uint64_t z, uint64_t fp, int maxval, uint64_t *values)
{
	fp = fingerprint(d, fp);
	uint64_t h = mix64(z) % d->size;
	int nval = 0;
	for (;;) {
		uint64_t e = d->slot[h];
		if (e == 0)
			return nval;
		if ((e & ~d->vmask) == fp) {
			if (nval == maxval)
				return -1;
			values[nval++] = (e & d->vmask) - 1;
		}
		if (++h == d->size)
			h = 0;
	}
}

#endif
