"""Bar chart for P3 layer probes from probe_summary_linear.json.

Example:
  python -m multisenge_utae.plot_probe_summary \\
    multisenge_utae/results/probe_c6_v0/probe_summary_linear.json \\
    --title "U-TAE P3 linear probes (val)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_probe_summary(summary: dict, out: Path, title: str, metric: str = "weighted_f1") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib not installed — on PARAM: python -m pip install --user matplotlib"
        ) from exc

    levels = list(summary.keys())
    label_map = {
        "weighted_f1": ("W-F1", "#16a34a"),
        "weighted_sensitivity": ("W-Sensitivity", "#2563eb"),
        "kappa": ("Kappa", "#9333ea"),
    }
    ylabel, color = label_map.get(metric, (metric, "#64748b"))
    values = [float(summary[level][metric]) for level in levels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(levels, values, color=color, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Encoder level")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot P3 probe summary bar chart")
    p.add_argument("summary_json", type=Path, help="probe_summary_linear.json")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: probe_summary_linear.png beside JSON)",
    )
    p.add_argument("--title", type=str, default="P3 layer probes (linear, val)")
    p.add_argument(
        "--metric",
        type=str,
        default="weighted_f1",
        choices=["weighted_f1", "weighted_sensitivity", "kappa", "accuracy"],
    )
    args = p.parse_args()

    if not args.summary_json.is_file():
        raise SystemExit(f"summary not found: {args.summary_json}")

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    out = args.out or args.summary_json.with_suffix(".png")
    plot_probe_summary(summary, out, args.title, metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
