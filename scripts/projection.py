"""Memory needed by the dictionary, and the nodes it takes, for larger n.

    python3 scripts/projection.py

Writes results/projection.csv. Only arithmetic: no run behind these numbers.
"""

import csv
import math
from pathlib import Path

NODE_GB = 96          # gros nodes of Grid'5000
USABLE = 0.85         # share of a node's memory left for the table
NODES_AVAILABLE = 50  # nodes used in the course runs

# bytes per key: entry size divided by load factor
COURSE = 12 / (1 / 1.125)
CURRENT = 8 / 0.8


def nodes_needed(total_bytes):
    return math.ceil(total_bytes / (NODE_GB * 1e9 * USABLE))


def main():
    rows = []
    for n in range(36, 45):
        keys = 2 ** n
        for version, per_key, passes in [("course", COURSE, 1)] + [
            ("current", CURRENT, k) for k in (1, 2, 4, 8, 16)
        ]:
            total = keys * per_key / passes
            nodes = nodes_needed(total)
            rows.append({
                "n": n,
                "version": version,
                "passes": passes,
                "dictionary_tb": round(total / 1e12, 3),
                "nodes_96gb": nodes,
                "fits_50_nodes": nodes <= NODES_AVAILABLE,
            })
    out = Path(__file__).resolve().parent.parent / "results" / "projection.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r["n"] in (38, 40, 42):
            print(r)


if __name__ == "__main__":
    main()
