/*
 * All-to-all exchange of (z, value) pairs between MPI ranks.
 *
 * The send buffer is already grouped by destination rank: send_count[r]
 * 64-bit words starting at send_displ[r]. The counts are exchanged with
 * MPI_Alltoall, then the data with MPI_Alltoallv. Both calls are
 * collective, so the messages of two consecutive rounds cannot be mixed up.
 */

#ifndef EXCHANGE_H
#define EXCHANGE_H

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	int p;
	int *send_count, *send_displ;
	int *recv_count, *recv_displ;
	uint64_t *recv;
	size_t recv_capacity;   /* in words */
} exchange_t;

static void exchange_init(exchange_t *e, int p)
{
	e->p = p;
	e->send_count = calloc(p, sizeof(int));
	e->send_displ = calloc(p, sizeof(int));
	e->recv_count = calloc(p, sizeof(int));
	e->recv_displ = calloc(p, sizeof(int));
	e->recv = NULL;
	e->recv_capacity = 0;
}

static void exchange_free(exchange_t *e)
{
	free(e->send_count);
	free(e->send_displ);
	free(e->recv_count);
	free(e->recv_displ);
	free(e->recv);
}

/* returns the number of words received, stored in e->recv */
static size_t exchange(exchange_t *e, const uint64_t *send)
{
	MPI_Alltoall(e->send_count, 1, MPI_INT, e->recv_count, 1, MPI_INT, MPI_COMM_WORLD);

	size_t total = 0;
	for (int r = 0; r < e->p; r++) {
		e->recv_displ[r] = (int)total;
		total += (size_t)e->recv_count[r];
	}
	if (total > e->recv_capacity) {
		free(e->recv);
		e->recv_capacity = total + total / 4;
		e->recv = malloc(e->recv_capacity * sizeof(uint64_t));
	}

	MPI_Alltoallv(send, e->send_count, e->send_displ, MPI_UINT64_T,
	              e->recv, e->recv_count, e->recv_displ, MPI_UINT64_T, MPI_COMM_WORLD);
	return total;
}

#endif
