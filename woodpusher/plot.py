"""Plot loss curves for runs, and the scaling plot across the ladder.

    uv run python -m woodpusher.plot            # all runs under runs/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(run_dir):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    rows = list(csv.DictReader(open(run_dir / "log.csv")))
    return meta, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [d for d in sorted(Path(args.runs_dir).glob("*"))
            if (d / "log.csv").exists() and (d / "run_meta.json").exists()]
    if not runs:
        print(f"no runs with log.csv found under {args.runs_dir}/")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    scaling = []
    for run_dir in runs:
        meta, rows = read_run(run_dir)
        label = f"{run_dir.name} ({meta['params'] / 1e6:.1f}M)"
        tokens = [int(r["tokens"]) for r in rows]
        train = [float(r["train_loss"]) for r in rows]
        ax.plot(tokens, train, alpha=0.4)
        val_pts = [(int(r["tokens"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
        if val_pts:
            ax.plot(*zip(*val_pts), marker="o", markersize=3, label=label)
            scaling.append((meta["params"], min(v for _, v in val_pts), run_dir.name))
    ax.set_xscale("log")
    ax.set_xlabel("training tokens")
    ax.set_ylabel("loss (train faint, val marked)")
    ax.legend()
    ax.set_title("woodpusher training curves")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    print(f"wrote {out / 'loss_curves.png'}")

    if len(scaling) >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        scaling.sort()
        params = [p for p, _, _ in scaling]
        losses = [l for _, l, _ in scaling]
        ax.plot(params, losses, marker="o")
        for p, l, name in scaling:
            ax.annotate(name, (p, l), textcoords="offset points", xytext=(6, 4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("parameters")
        ax.set_ylabel("best val loss")
        ax.set_title("your scaling law")
        fig.tight_layout()
        fig.savefig(out / "scaling.png", dpi=150)
        print(f"wrote {out / 'scaling.png'}")


if __name__ == "__main__":
    main()
