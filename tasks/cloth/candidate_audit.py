"""Export every candidate outcome and model ranking for both task banks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from experiment import TEST, task
from folding import geometry, metrics
from learn import FoldResponse, get_path, plan, score


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = cli.parse_args()
    torch.set_num_threads(1)
    root = args.root
    model = FoldResponse(root / "fold-response.npz").eval()
    audits = {}
    for folder, seed0 in [("test", 50100), ("confirm", 50300)]:
        rows = []
        for i in range(TEST):
            seed, land, actions = task(i, seed0)
            g = geometry(seed)
            _, _qhat, pm = plan(model, g, actions, land)
            candidates = []
            for j, action in enumerate(actions):
                with np.load(get_path(root / folder, i, j)) as z:
                    q = z["q"]
                    truth = metrics(q, g, land)
                    diag = {
                        k: float(z[k])
                        for k in ("max_edge_stretch", "min_z_m", "max_table_candidates")
                    }
                candidates.append(
                    {
                        "index": j,
                        "action": action.tolist(),
                        "teacher": truth,
                        "teacher_score": float(score(truth, g)),
                        "predicted": pm[j],
                        "predicted_score": float(score(pm[j], g)),
                        "teacher_diagnostics": diag,
                    }
                )
            rows.append(
                {
                    "seed": seed,
                    "geometry": g.tolist(),
                    "landing": land,
                    "candidates": candidates,
                }
            )
        audits[folder] = rows
    (root / "all-candidates.json").write_text(
        json.dumps(audits, indent=2, allow_nan=False) + "\n"
    )
    for key, cases in audits.items():
        print(
            key,
            "has at least one feasible action",
            sum(any(x["teacher"]["success"] for x in r["candidates"]) for r in cases),
        )
    # Directional derivative check of the open-loop response with respect to its prescribed action.
    geom = torch.from_numpy(geometry(50300)).double()[None, :]
    action = torch.tensor(
        [[0.19, 0.42, 0.039]], dtype=torch.float64, requires_grad=True
    )
    model = model.double()
    direction = torch.tensor([[0.4, -0.27, 0.025]], dtype=torch.float64)

    def value(a):
        q = model(geom, a)
        return q[:, -1, :, 2].square().mean() + q[:, -1, :, 0].square().mean()

    exact = (torch.autograd.grad(value(action), action)[0] * direction).sum().item()
    h = 1e-4
    fd = (
        (value(action + h * direction) - value(action - h * direction)) / (2 * h)
    ).item()
    (root / "differentiability.json").write_text(
        json.dumps(
            {
                "exact": exact,
                "finite_difference": fd,
                "h": h,
                "relative_error": abs(fd - exact) / max(abs(exact), 1e-10),
                "scope": "Gradient of model output, independent of teacher.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
