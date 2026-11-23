#!/bin/bash
# Course version against the current version on one machine, then the cost
# of splitting the dictionary into passes.
# Writes results/local_benchmark.csv and results/local_passes.csv.
#
#   make && bench/local_benchmark.sh
#
# The course version is built from course/ into build/mitm_course.
set -eu
cd "$(dirname "$0")/.."
MPIRUN="mpirun --bind-to none ${MPIRUN_FLAGS:-}"
REPS=${REPS:-3}

mkdir -p build
mpicc -O3 -march=native -fopenmp course/mitm_mpi_2_block.c -o build/mitm_course

out=results/local_benchmark.csv
echo "version,n,ranks,threads,rep,seconds,dictionary_mb_per_rank,found" > "$out"

for n in 22 24 26; do
	read -r c0 c1 k1 k2 <<< "$(build/gen_instance "$n" "$n")"
	for cfg in "2 1" "1 2"; do
		set -- $cfg
		np=$1; th=$2
		for rep in $(seq "$REPS"); do
			log=$(OMP_NUM_THREADS=$th $MPIRUN -np "$np" build/mitm_course --n "$n" --C0 "$c0" --C1 "$c1" < /dev/null)
			t=$(grep 'Time taken' <<< "$log" | sed 's/.*= //; s/ seconds//' | sort -n | tail -1)
			mb=$(awk -v n="$n" -v p="$np" 'BEGIN { printf "%.1f", 1.125 * 2^n / p * 12 / 1e6 }')
			ok=$(grep -c "Solution found: ($k1, $k2)" <<< "$log" || true)
			echo "course,$n,$np,$th,$rep,$t,$mb,$ok" >> "$out"

			log=$(OMP_NUM_THREADS=$th $MPIRUN -np "$np" build/mitm --n "$n" --C0 "$c0" --C1 "$c1" < /dev/null)
			t=$(grep '^time:' <<< "$log" | awk '{print $2}')
			mb=$(awk -v n="$n" -v p="$np" 'BEGIN { printf "%.1f", (int(2^n / p / 0.8) + 4096) * 8 / 1e6 }')
			ok=$(grep -c "solution: $k1 $k2" <<< "$log" || true)
			echo "current,$n,$np,$th,$rep,$t,$mb,$ok" >> "$out"
			echo "n=$n ranks=$np threads=$th rep=$rep done"
		done
	done
done

out=results/local_passes.csv
echo "n,ranks,threads,passes,rep,seconds,evaluation,exchange,insertion,probing,dictionary_mb_per_rank,found" > "$out"
n=26
read -r c0 c1 k1 k2 <<< "$(build/gen_instance "$n" "$n")"
for k in 1 2 4 8; do
	for rep in $(seq "$REPS"); do
		log=$(OMP_NUM_THREADS=1 $MPIRUN -np 2 build/mitm --n "$n" --C0 "$c0" --C1 "$c1" --passes "$k" < /dev/null)
		t=$(grep '^time:' <<< "$log" | sed 's/[(),;]/ /g' | awk '{print $2","$5","$7","$9","$11}')
		mb=$(grep 'dictionary per rank' <<< "$log" | awk '{print $4}')
		ok=$(grep -c "solution: $k1 $k2" <<< "$log" || true)
		echo "$n,2,1,$k,$rep,$t,$mb,$ok" >> "$out"
		echo "passes=$k rep=$rep done"
	done
done
