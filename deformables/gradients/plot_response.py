"""Plot force sensitivity of the cable tip and cloth free edge for a fixed unseen seed."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gradients.local_fit import forward_tangent, initial_model
from native_rollouts import decode

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    parent = initial_model(ROOT / "weights/modal-paired.pt", "cpu").eval()
    update = initial_model(
        ROOT / "gradients/results/local/autodiff-271.pt", "cpu"
    ).eval()
    fig, axes = plt.subplots(
        2, 1, figsize=(8.4, 5.2), sharex=True, layout="constrained"
    )
    for family, ax in zip(("cable", "cloth"), axes, strict=True):
        with np.load(ROOT / f"gradients/data/probe/{family}-42760.npz") as z:
            data = dict(z)
        rest, free = data["rest"], data["free"].reshape(-1)
        direction = data["direction"]
        vector = direction[np.linalg.norm(direction, axis=1).argmax()]
        vector = vector / np.linalg.norm(vector)
        nx = int(data["static"][3]) + 1
        end = np.arange(nx - 1, len(rest), nx)
        t = 0.02 * np.arange(1, 301)

        def projected(x, end=end, vector=vector):
            return (x[:, end] @ vector).mean(axis=1) * 1000

        actual = (data["native"][3].astype(float) - data["native"][1]) / 0.5
        ax.plot(
            t,
            projected(actual),
            color="#123750",
            label="Newton central difference",
            lw=2.0,
        )
        inputs = [
            torch.from_numpy(data[k][None])
            for k in ("q0", "v0", "static", "u", "direction")
        ]
        for model, label, color, dash in (
            (parent, "Original shared model", "#bb6135", "--"),
            (update, "Derivative fine-tune", "#00786c", "-"),
        ):
            slope = forward_tangent(model, *inputs)[0].numpy()
            native = decode(slope, data["static"], rest, free) - rest
            ax.plot(t, projected(native), dash, color=color, label=label, lw=1.45)
        ax.grid(axis="y", alpha=0.18)
        ax.set_ylabel("mm / (m/s²)")
        ax.set_title(
            "Cable tip" if family == "cable" else "Cloth free edge",
            loc="left",
            fontsize=11,
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Simulated time (s)")
    fig.savefig(ROOT.parent / "figures/force-sensitivity.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
