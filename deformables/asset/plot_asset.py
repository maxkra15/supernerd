"""Plot test rollout error against calibration sample count for both materials."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).parent
    fresh = json.loads((root / "results/fresh.json").read_text())["results"]
    verified = json.loads((root / "results/verified.json").read_text())["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained", sharey=True)
    palette = {"cable": "#08686e", "cloth": "#a24335"}
    for ax, family in zip(axes, ("cable", "cloth"), strict=True):
        data = fresh[family]
        sizes = [2, 4, 8, 12]
        line = [data["fit"][str(n)]["rms_mm"] for n in sizes]
        nominal = data["nominal"]["rms_mm"]
        ax.axhline(nominal, color="#4f5961", linestyle="--", label="Nominal replay")
        ax.plot(sizes, line, color=palette[family], marker="o", lw=2.4, label="Gaussian RBF")
        ax.scatter(12, verified[family]["phase"]["rms_mm"], color="#282b3c", marker="x", s=78,
                   linewidth=2, label="Phase-linear, 12 programs", zorder=3)
        ax.set(title=family.capitalize(), xlabel="Calibration force programs", xticks=sizes, ylim=(0, 27))
        ax.grid(axis="y", color="#dfe4e3", lw=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Native-mesh position RMSE (mm)")
    fig.savefig(root / "results/calibration-curve.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
