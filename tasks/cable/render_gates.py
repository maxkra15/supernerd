"""Render the first fixed cable-gate task from measured Newton and model trajectories."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--example", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    with np.load(args.example) as source:
        time, targets = source["times"], source["target"]
        teacher, surrogate = source["supernerd_mixed_replay"], source["supernerd_mixed"]
    fig = plt.figure(figsize=(8.4, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.7, 1.1))
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    clock = fig.add_subplot(grid[1, :])
    colors = ["#19435b", "#bf6139"]
    data = [teacher, surrogate]
    artists = []

    def projected(p):
        return p[..., 0] + 0.60 * p[..., 1], p[..., 2] + 0.30 * p[..., 1]

    for ax, name, color, positions in zip(
        axes, ["Newton replay", "SuperNeRD forecast"], colors, data, strict=True
    ):
        ax.set_aspect("equal")
        x, z = projected(np.concatenate(data, axis=0))
        ax.set_xlim(float(x.min() - 0.035), float(x.max() + 0.035))
        ax.set_ylim(float(z.min() - 0.045), float(z.max() + 0.045))
        ax.set_axis_off()
        ax.set_title(name)
        xx, zz = projected(positions[0, :2])
        ax.scatter(xx, zz, marker="s", s=20, color="#435253")
        (line,) = ax.plot([], [], color=color, lw=3, marker=".", ms=4)
        (dot,) = ax.plot([], [], marker="o", color=color, ms=6)
        artists.append((line, dot))
    clock.set(
        xlim=(0, time[-1] + 0.04),
        xlabel="Simulated time (s)",
        ylabel="Lateral tip position (mm)",
    )
    clock.axhline(0, color="#8b9995", lw=0.7)
    for t, y in zip((1.10, 2.40), 1000 * targets, strict=True):
        clock.errorbar(
            t, y, yerr=10.0, fmt="s", color="#2c7558", capsize=5, ms=3, zorder=4
        )
    dots = []
    for d, color, name in zip(data, colors, ("Newton", "SuperNeRD"), strict=True):
        clock.plot(
            time,
            d[:, -1, 1] * 1000,
            color=color,
            ls="-" if name == "Newton" else "--",
            lw=1.5,
            label=name,
        )
        dots.append(clock.plot([], [], "o", color=color, ms=5)[0])
    clock.errorbar(
        [], [], yerr=[], fmt="s", color="#2c7558", capsize=4, ms=3, label="Gate ±10 mm"
    )
    clock.spines[["top", "right"]].set_visible(False)
    clock.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    cursor = clock.axvline(0, color="#475455", alpha=0.55, lw=1)

    def animate(i):
        for d, (line, dot), stamp in zip(data, artists, dots, strict=True):
            x, z = projected(d[i])
            line.set_data(x, z)
            dot.set_data(x[-1:], z[-1:])
            stamp.set_data([time[i]], [d[i, -1, 1] * 1000])
        cursor.set_xdata([time[i], time[i]])

    animate(55)
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    indices = np.r_[np.arange(0, len(time), 2), np.full(6, len(time) - 1)]
    FuncAnimation(fig, animate, frames=indices, interval=40).save(
        args.output, writer=PillowWriter(fps=25), dpi=105
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
