"""
Plot the Pareto front from one or more ECF milestone XML files.

Each milestone's <Individual> entries carry a <MOFitness value1="..." value2="..."/>
where the two values are the *negated* objectives (ECF's MOFitnessMin convention):
    value1 = -similarity        (we maximize similarity, fed to ECF as -similarity)
    value2 = -compression_ratio (same)

This script:
  1. parses one or more milestone files,
  2. un-negates,
  3. computes the non-dominated set under (sim ↑, comp ↑) maximization,
  4. plots compression on the x-axis and similarity on the y-axis (paper Fig. 3 orientation).

Usage:
    python visualization/plot_pareto.py milestone-5h.txt
    python visualization/plot_pareto.py milestone-5h.txt -o visualization/pareto_5h.png
    python visualization/plot_pareto.py milestone-5h.txt milestone.txt -o pareto_compare.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: works on cluster nodes too
import matplotlib.pyplot as plt


_FITNESS_RE = re.compile(r'MOFitness\s+value1="([^"]+)"\s+value2="([^"]+)"')


def parse_milestone(path: Path) -> list[tuple[float, float]]:
    """Return list of (similarity, compression) pairs after un-negating ECF's MOFitnessMin."""
    text = path.read_text()
    out: list[tuple[float, float]] = []
    for m in _FITNESS_RE.finditer(text):
        sim = -float(m.group(1))
        comp = -float(m.group(2))
        out.append((sim, comp))
    return out


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated subset under (sim, comp) maximization. Returns sorted by compression."""
    uniq = sorted(set(points))
    front = []
    for p in uniq:
        s, c = p
        dominated = False
        for s2, c2 in uniq:
            if (s2, c2) == p:
                continue
            if s2 >= s and c2 >= c and (s2 > s or c2 > c):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda x: x[1])
    return front


def plot(milestones: list[Path], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, path in enumerate(milestones):
        all_pts = parse_milestone(path)
        if not all_pts:
            print(f"warning: no fitness pairs found in {path}", file=sys.stderr)
            continue

        front = pareto_front(all_pts)
        # split front into NaN-clamped (sim == -1.0) and proper individuals
        proper = [(s, c) for s, c in front if s > -0.999]
        clamped = [(s, c) for s, c in front if s <= -0.999]

        c_main = colors[i % len(colors)]
        label = path.stem

        # dominated population (faint background)
        dominated = [p for p in all_pts if p not in set(front)]
        if dominated:
            ds, dc = zip(*dominated)
            ax.scatter(dc, ds, s=12, alpha=0.15, color=c_main)

        # non-dominated front (proper)
        if proper:
            xs = [c for _, c in proper]
            ys = [s for s, _ in proper]
            ax.plot(xs, ys, "-o", color=c_main, label=f"{label} ({len(proper)} pts)",
                    markersize=6, linewidth=1.5)

        # NaN-clamped individuals on the front
        if clamped:
            xs = [c for _, c in clamped]
            ys = [0.0] * len(xs)  # plot at y=0 as "below the chart" markers
            ax.scatter(xs, ys, marker="x", s=50, color=c_main, alpha=0.6,
                       label=f"{label} NaN-clamped (×{len(clamped)})")

    ax.set_xlabel("Compression ratio  (layers removed / 32)")
    ax.set_ylabel("Module-wise similarity")
    ax.set_title("GeLaCo Pareto front")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("milestones", nargs="+", type=Path,
                   help="One or more ECF milestone XML files.")
    p.add_argument("-o", "--out", type=Path, default=Path("pareto.png"),
                   help="Output image path (default: pareto.png).")
    args = p.parse_args()

    for m in args.milestones:
        if not m.exists():
            print(f"error: {m} does not exist", file=sys.stderr)
            return 2

    plot(args.milestones, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
