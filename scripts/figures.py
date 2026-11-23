"""Figures of the README, from the CSV files in results/.

    python3 scripts/figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

DARK, MID, LIGHT, ACCENT, GREY = "#1f3b57", "#4f7ca8", "#a9c3dd", "#c0504d", "#8c8c8c"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "figure.dpi": 150,
})


def memory():
    d = pd.read_csv(RESULTS / "projection.csv")
    capacity = 50 * 96 * 0.85 / 1000  # TB usable on 50 gros nodes
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    curves = [
        ("course", 1, "course version", GREY, "--"),
        ("current", 1, "current, 1 pass", DARK, "-"),
        ("current", 4, "current, 4 passes", MID, "-"),
        ("current", 16, "current, 16 passes", LIGHT, "-"),
    ]
    for version, passes, label, color, style in curves:
        s = d[(d.version == version) & (d.passes == passes)]
        ax.plot(s.n, s.dictionary_tb, style, color=color, marker="o", ms=3, label=label)
    ax.axhline(capacity, color=ACCENT, lw=1)
    ax.text(36.1, capacity * 1.15, "memory of 50 gros nodes", color=ACCENT, fontsize=9)
    ax.scatter([38], [d[(d.version == "course") & (d.n == 38)].dictionary_tb], s=60,
               facecolors="none", edgecolors=ACCENT, zorder=5)
    ax.annotate("n = 38, solved in the course", (38, 3.7), xytext=(36.1, 20), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.set_yscale("log")
    ax.set_xlabel("n (bits)")
    ax.set_ylabel("dictionary, all ranks (TB)")
    ax.set_title("Memory needed by the dictionary", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "memory.png")


def local():
    d = pd.read_csv(RESULTS / "local_benchmark.csv")
    m = d.groupby(["n", "ranks", "threads", "version"]).seconds.median().unstack("version")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=False)
    for ax, (ranks, threads) in zip(axes, [(2, 1), (1, 2)]):
        s = m.xs((ranks, threads), level=("ranks", "threads"))
        x = range(len(s))
        ax.bar([i - 0.2 for i in x], s["course"], 0.4, color=LIGHT, label="course version")
        ax.bar([i + 0.2 for i in x], s["current"], 0.4, color=DARK, label="current")
        for i, (c, k) in enumerate(zip(s["course"], s["current"])):
            ax.text(i + 0.2, k, f"{c / k:.1f}x", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x), [f"n = {n}" for n in s.index])
        ax.set_title(f"{ranks} rank{'s' if ranks > 1 else ''} x {threads} thread{'s' if threads > 1 else ''}",
                     fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("seconds (median of 3)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("One machine, two cores", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "local.png")


def passes():
    d = pd.read_csv(RESULTS / "local_passes.csv")
    parts = ["evaluation", "exchange", "insertion", "probing"]
    m = d.groupby("passes")[parts + ["seconds"]].median()
    # time not covered by the four timers: grouping pairs by owner, waiting at collectives
    m["other"] = m["seconds"] - m[parts].sum(axis=1)
    parts = parts + ["other"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bottom = [0.0] * len(m)
    for part, color in zip(parts, [DARK, GREY, MID, LIGHT, "#d9d9d9"]):
        ax.bar([str(k) for k in m.index], m[part], 0.6, bottom=bottom, color=color, label=part)
        bottom = [b + v for b, v in zip(bottom, m[part])]
    mem = d.groupby("passes").dictionary_mb_per_rank.first()
    for i, (k, top) in enumerate(zip(m.index, bottom)):
        ax.text(i, top, f"{mem[k]:.0f} MB", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("passes (n = 26, 2 ranks; label: dictionary per rank)")
    ax.set_ylabel("seconds (median of 3)")
    ax.set_title("Trading memory for computation", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    ax.set_ylim(0, max(bottom) * 1.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "passes.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    memory()
    local()
    passes()
