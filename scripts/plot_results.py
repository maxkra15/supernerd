"""Generate vector figures for the SuperNeRD manuscript from published evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"pathwise": "#007076", "ppo": "#975225", "newton_ppo": "#55488b"}
NAMES = {"pathwise": "Backpropagation", "ppo": "PPO · learned", "newton_ppo": "PPO · Newton"}
HORIZONS = (1, 15, 100, 200)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#172436",
            "axes.edgecolor": "#9babb5",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#334959",
            "ytick.color": "#334959",
            "text.color": "#172436",
            "grid.color": "#d5e0e6",
            "grid.alpha": 0.7,
            "svg.fonttype": "none",
            "svg.hashsalt": "supernerd",
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15, metadata={"Date": None})
    content = "\n".join(line.rstrip() for line in path.read_text().splitlines())
    path.write_text(content + "\n", encoding="utf-8")
    plt.close(fig)


def plot_learning(arms: dict, output: Path, name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex="col", layout="constrained")
    for arm, record in arms.items():
        curves = [seed["curve"] for seed in record["seeds"]]
        options = {"color": COLORS[arm], "label": NAMES[arm]}
        for col, (x_key, scale) in enumerate((("simulator_steps", 1e3), ("training_s", 1))):
            for row, (metric, y_scale) in enumerate((("mean_native_cost", 1), ("success_rate", 100))):
                ax = axes[row, col]
                for curve in curves:
                    x = [p[x_key] / scale for p in curve]
                    y = [p["newton"][metric] * y_scale for p in curve]
                    ax.plot(x, y, color=COLORS[arm], alpha=0.22, lw=0.9)
                # Budget grids can differ in duration across seeds. Interpolate within
                # their common range so averages always use the complete seed set.
                begin = max(curve[0][x_key] / scale for curve in curves)
                end = min(curve[-1][x_key] / scale for curve in curves)
                grid = np.linspace(begin, end, 150)
                mean = np.mean(
                    [
                        np.interp(grid, [p[x_key] / scale for p in curve], [p["newton"][metric] * y_scale for p in curve])
                        for curve in curves
                    ],
                    axis=0,
                )
                ax.plot(grid, mean, lw=2.1, **options)
                ax.grid(axis="y")
    axes[0, 0].set_ylabel("Mean Newton episode cost")
    axes[1, 0].set_ylabel("Newton success (%)")
    axes[1, 0].set_ylim(-3, 103)
    axes[1, 1].set_ylim(-3, 103)
    axes[1, 0].set_xlabel("Training transitions (thousands)")
    axes[1, 1].set_xlabel("Synchronized policy training (s)")
    axes[0, 1].legend(frameon=False, loc="best")
    save(fig, output / name)


def plot_rollouts(rows: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55), sharey=True, layout="constrained")
    colors = ("#007076", "#975225", "#55488b", "#5f6b76", "#60812c")
    for ax, (cohort, series) in zip(axes, rows.items(), strict=True):
        for i, (name, points) in enumerate(series.items()):
            ax.plot(HORIZONS, [points[str(h)] for h in HORIZONS], color=colors[i], lw=1.7,
                    marker=("o", "x", "s", "^", "D")[i], label=name)
        ax.set(xscale="log", yscale="log", xticks=HORIZONS, xlabel="Recursive horizon (steps)",
               title=f"{cohort.capitalize()} trajectories")
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.grid(axis="y", which="both")
    axes[0].set_ylabel("Prefix NRMSE")
    axes[0].legend(frameon=False, loc="center left", fontsize=7.5)
    save(fig, output / "rollout-errors.svg")


def plot_gradients(payload: dict, output: Path, name: str) -> None:
    results = payload["results"]
    horizons = payload["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), layout="constrained")
    places = np.arange(len(horizons))
    rng = np.random.default_rng(43)
    for cohort, label, offset, color in (
        ("near_upright", "Near upright", -0.12, "#007076"),
        ("wide_state", "Wide state", 0.12, "#975225"),
    ):
        for idx, horizon in enumerate(horizons):
            row = results[str(horizon)][cohort]
            cosines = row["case_gradient_cosine"]
            reductions = np.asarray(row["case_teacher_relative_cost_reduction"]) * 100
            jitter = rng.uniform(-0.055, 0.055, len(cosines))
            for ax, values in zip(axes, (cosines, reductions), strict=True):
                pos = places[idx] + offset
                ax.scatter(pos + jitter, values, s=11, color=color, alpha=0.45, edgecolor="none")
                ax.scatter(pos, np.median(values), marker="D", s=29, color=color,
                           label=label if idx == 0 else None, edgecolor="white", linewidth=0.55, zorder=3)
    axes[0].set_ylabel("Gradient cosine with Newton")
    axes[1].set_ylabel("Newton cost reduction (%)")
    for ax in axes:
        ax.axhline(0, color="#8298a3", lw=0.9)
        ax.set(xticks=places, xticklabels=horizons, xlabel="Action horizon (steps)")
        ax.grid(axis="y")
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/figure-data.json"))
    parser.add_argument("--gradients", type=Path, default=Path("results/action-gradients.json"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    bundle = json.loads(args.metrics.read_text())
    plot_learning(bundle["policy_arms"], args.output, "policy-learning.svg")
    plot_learning(bundle["extended_policy_arms"], args.output, "extended-policy-learning.svg")
    plot_rollouts(bundle["rollout_series"], args.output)
    plot_gradients(json.loads(args.gradients.read_text()), args.output, "action-gradients.svg")
    structured = args.gradients.with_name("action-gradients-structured.json")
    plot_gradients(json.loads(structured.read_text()), args.output, "action-gradients-structured.svg")


if __name__ == "__main__":
    main()
