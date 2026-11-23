#!/bin/bash
# Correctness tests on small instances with a known solution.
# Each line: n, MPI ranks, OpenMP threads, passes, block size (log2), seed.
# With n = 12 and 17 ranks, the slices have 240 or 241 keys: with blocks of 16
# keys, ranks would need 15 or 16 rounds if each counted its own.
set -u
cd "$(dirname "$0")/.."
MPIRUN="mpirun --oversubscribe ${MPIRUN_FLAGS:-}"
fail=0
while read -r n np th k b seed; do
	read -r c0 c1 k1 k2 <<< "$(build/gen_instance "$n" "$seed")"
	out=$(OMP_NUM_THREADS=$th $MPIRUN -np "$np" build/mitm --n "$n" --C0 "$c0" --C1 "$c1" \
	      --passes "$k" --block "$b" < /dev/null 2>&1)
	if grep -q "solution: $k1 $k2" <<< "$out"; then
		echo "ok    n=$n ranks=$np threads=$th passes=$k"
	else
		echo "FAIL  n=$n ranks=$np threads=$th passes=$k"
		fail=1
	fi
done <<'CASES'
8 1 1 1 4 1
12 7 1 1 6 2
16 1 2 1 10 3
16 2 2 1 12 4
18 3 1 1 12 5
18 4 1 3 10 6
20 2 2 4 14 7
20 5 1 2 12 8
22 2 1 1 20 9
22 3 1 8 16 10
12 17 1 1 4 11
CASES
exit $fail
