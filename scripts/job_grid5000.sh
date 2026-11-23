#!/bin/bash
# Batch job for the OAR scheduler of Grid'5000.
#
#   oarsub -l nodes=50,walltime=01:00:00 ./scripts/job_grid5000.sh
#
# Edit the instance and the layout below. RANKS_PER_NODE x THREADS should not
# exceed the number of cores of a node (36 on the gros cluster).

N=38
C0=a9bf4a972ee54312
C1=1831ee7f563077ed
PASSES=1
RANKS_PER_NODE=18
THREADS=2

set -eu
cd "$(dirname "$0")/.."
make -s

NODES=$(sort -u "$OAR_NODEFILE" | wc -l)
NP=$((NODES * RANKS_PER_NODE))
export OMP_NUM_THREADS=$THREADS

echo "nodes=$NODES ranks=$NP threads=$THREADS n=$N passes=$PASSES"
mpirun -machinefile "$OAR_NODEFILE" -np "$NP" \
	--map-by "ppr:$RANKS_PER_NODE:node" --bind-to none \
	-x OMP_NUM_THREADS \
	build/mitm --n "$N" --C0 "$C0" --C1 "$C1" --passes "$PASSES"
