"""Measure how different a checkpoint's force JVP is from its own finite secant.

Run on the original training pairs for six seconds from their common release,
with CPU float64 arithmetic to isolate genuine nonlinear curvature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gradients.force_gradient_trial import midpoint_jvp
from modal_forward import ModalForward, load


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--weights", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    a = load(args.data / "train.npz", "cpu")
    b = load(args.data / "train-pulses.npz", "cpu")
    model = ModalForward().double().eval()
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)
    )
    rows = []
    with torch.no_grad():
        for i in range(0, len(a["seed"]), 8):
            use = slice(i, i + 8)
            q, v, s = (
                a["q"][use, 0].double(),
                a["v"][use, 0].double(),
                a["static"][use].double(),
            )
            u, w = a["u"][use].double(), b["u"][use].double()
            j = midpoint_jvp(model, q, v, s, u, w)
            sec = model(q, v, s, u) - model(q, v, s, w)
            target = (a["q"][use, 1:] - b["q"][use, 1:]).double()

            def norm(x):
                return x.flatten(1).norm(dim=1)

            for f, seed, gap, secsize, tsize in zip(
                a["family"][use],
                a["seed"][use],
                norm(j - sec),
                norm(sec),
                norm(target),
                strict=True,
            ):
                rows.append(
                    {
                        "family": str(f),
                        "seed": int(seed),
                        "gap": float(gap),
                        "secant": float(secsize),
                        "truth": float(tsize),
                    }
                )
    result = {}
    for fam in ("cable", "cloth"):
        sub = [r for r in rows if r["family"] == fam]
        gap, secant, truth = ([r[k] for r in sub] for k in ("gap", "secant", "truth"))
        x = 100 * np.asarray(gap) / np.asarray(secant)
        result[fam] = {
            "count": len(sub),
            "median_pct_model_secant": float(np.median(x)),
            "p90_pct_model_secant": float(np.percentile(x, 90)),
            "pooled_pct_model_secant": float(
                100 * np.linalg.norm(gap) / np.linalg.norm(secant)
            ),
            "pooled_pct_newton_response": float(
                100 * np.linalg.norm(gap) / np.linalg.norm(truth)
            ),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
