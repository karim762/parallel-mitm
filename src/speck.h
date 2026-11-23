/* Speck64/128 block cipher (Beaulieu et al., 2013), as used to build the instances. */

#ifndef SPECK_H
#define SPECK_H

#include <stdint.h>

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))

#define ER32(x, y, k) (x = ROTR32(x, 8), x += y, x ^= k, y = ROTL32(y, 3), y ^= x)
#define DR32(x, y, k) (y ^= x, y = ROTR32(y, 3), x ^= k, x -= y, x = ROTL32(x, 8))

/* the two plaintexts of the problem */
static const uint32_t SPECK_P0[2] = {0, 0};
static const uint32_t SPECK_P1[2] = {0xffffffff, 0xffffffff};

static inline void speck_key_schedule(const uint32_t K[4], uint32_t rk[27])
{
	uint32_t i, D = K[3], C = K[2], B = K[1], A = K[0];
	for (i = 0; i < 27;) {
		rk[i] = A; ER32(B, A, i++);
		rk[i] = A; ER32(C, A, i++);
		rk[i] = A; ER32(D, A, i++);
	}
}

static inline void speck_encrypt(const uint32_t pt[2], uint32_t ct[2], const uint32_t rk[27])
{
	ct[0] = pt[0];
	ct[1] = pt[1];
	for (int i = 0; i < 27;)
		ER32(ct[1], ct[0], rk[i++]);
}

static inline void speck_decrypt(uint32_t pt[2], const uint32_t ct[2], const uint32_t rk[27])
{
	pt[0] = ct[0];
	pt[1] = ct[1];
	for (int i = 26; i >= 0;)
		DR32(pt[1], pt[0], rk[i--]);
}

#endif
