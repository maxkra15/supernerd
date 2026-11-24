"""Fit and evaluate a differentiable open-loop cloth fold response model and shot planner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from experiment import TEST, task
from folding import DT, FRAMES, HOLD_HEIGHT, NX, NY, geometry, metrics, target
from torch import nn

# Kernel distance is defined in SI units, fixed before opening the evaluation bank.
SCALES = np.array([0.27, 0.20, 725.0, 0.001, 0.85, 0.38, 0.27, 0.065], dtype=np.float32)
CENTER = np.array(
    [0.53, 0.345, 500.0, 0.0004, 0.515, 0.165, 0.43, 0.045], dtype=np.float32
)
KNOTS = np.arange(0, FRAMES + 1, 4)
RIDGE = 0.001


def rest_shape(g):
    x, y = np.meshgrid(
        np.linspace(0, g[0], NX + 1), np.linspace(-g[1] / 2, g[1] / 2, NY + 1)
    )
    return np.stack([x, y, np.full_like(x, HOLD_HEIGHT)], -1).reshape(-1, 3).astype("f")


def fit(folder, out):
    files = sorted(folder.glob("*.npz"))
    assert len(files) == 200
    x, y = [], []
    for path in files:
        with np.load(path) as z:
            g, a, q = z["g"], z["a"], z["q"]
            x.append((np.r_[g, a] - CENTER) / SCALES)
            y.append(((q[KNOTS] - q[:1]) / g[[0, 1, 0]]).reshape(-1))
    x = np.stack(x).astype(np.float64)
    y = np.stack(y).astype(np.float64)
    distance = np.square(x[:, None] - x).sum(-1)
    k = np.exp(-0.5 * distance) + RIDGE * np.eye(len(x))
    linear = np.concatenate([np.ones([len(x), 1]), x], 1)
    augmented = np.block([[k, linear], [linear.T, np.zeros([linear.shape[1]] * 2)]])
    coefficients = np.linalg.solve(
        augmented, np.r_[y, np.zeros([linear.shape[1], y.shape[1]])]
    )
    np.savez_compressed(out, centers=x.astype("f"), coef=coefficients.astype("f"))


class FoldResponse(nn.Module):
    """Predict 81 native meshes for a fixed-release two-grasp folding program."""

    def __init__(self, path):
        super().__init__()
        with np.load(path) as z:
            self.register_buffer("centers", torch.from_numpy(z["centers"]))
            self.register_buffer("coef", torch.from_numpy(z["coef"]))
        self.register_buffer("scales", torch.from_numpy(SCALES.copy()))
        self.register_buffer("center", torch.from_numpy(CENTER.copy()))
        self.register_buffer("unit", torch.from_numpy(rest_shape([1.0, 1.0])))
        t = torch.arange(FRAMES + 1) / 4
        self.register_buffer("left", t.long().clamp_max(len(KNOTS) - 2))
        self.register_buffer("frac", (t - self.left)[None, :, None, None])

    def forward(self, geom, action):
        x = (torch.cat([geom, action], -1) - self.center) / self.scales
        k = torch.exp(-0.5 * (x[:, None] - self.centers).square().sum(-1))
        phi = torch.cat([k, torch.ones_like(x[:, :1]), x], -1)
        knots = (phi @ self.coef).reshape(len(x), len(KNOTS), -1, 3)
        free = (1 - self.frac) * knots[:, self.left] + self.frac * knots[
            :, self.left + 1
        ]
        length = geom[:, None, None, [0, 1, 0]]
        rest = (
            self.unit
            * torch.cat([geom[:, :2], torch.ones_like(geom[:, :1])], -1)[:, None, :]
        )
        return rest[:, None] + free * length


def score(m, g):
    return (
        m["shape_error_percent"] / 100
        + max(0, m["final_height_p95_m"] - 0.035) / g[0]
        + max(0, m["corner_error_percent"] / 100 - 0.10)
    )


def contact_audit(q, g, landing):
    """Independent endpoint topology checks using nonadjacent segment-triangle crossings."""
    last = q[-1].astype(np.float64)
    ij = np.arange((NX + 1) * (NY + 1)).reshape(NY + 1, NX + 1)
    a, b, c, d = ij[:-1, :-1], ij[:-1, 1:], ij[1:, :-1], ij[1:, 1:]
    tri = np.concatenate(
        [np.stack((a, b, d), -1).reshape(-1, 3), np.stack((a, d, c), -1).reshape(-1, 3)]
    )
    edges = np.unique(
        np.sort(
            np.concatenate([tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]]]), axis=-1
        ),
        axis=0,
    )
    nonneighbor = (edges[:, None, :, None] != tri[None, :, None, :]).all(axis=(2, 3))
    p0, p1 = last[edges].transpose(1, 0, 2)
    t0, t1, t2 = last[tri].transpose(1, 0, 2)
    # AABB rejection keeps exact crossing tests small.
    bb_l = np.maximum(np.minimum(p0, p1)[:, None], np.minimum(np.minimum(t0, t1), t2))
    bb_h = np.minimum(np.maximum(p0, p1)[:, None], np.maximum(np.maximum(t0, t1), t2))
    ei, ti = np.nonzero(nonneighbor & (bb_l <= bb_h).all(-1))
    ray = (p1 - p0)[ei]
    e1 = (t1 - t0)[ti]
    e2 = (t2 - t0)[ti]
    h = np.cross(ray, e2)
    dot = lambda u, v: np.einsum("ij,ij->i", u, v)
    det = dot(e1, h)
    inv = np.divide(1, det, out=np.zeros_like(det), where=abs(det) > 1e-13)
    s = p0[ei] - t0[ti]
    v1 = inv * dot(s, h)
    cross = np.cross(s, e1)
    v2 = inv * dot(ray, cross)
    dist = inv * dot(e2, cross)
    eps = 1e-7
    crossed = (
        (abs(det) > 1e-13)
        & (v1 > eps)
        & (v2 > eps)
        & (v1 + v2 < 1 - eps)
        & (dist > eps)
        & (dist < 1 - eps)
    )
    # Vertical order across the target's folded overlap. Interpolate bottom layer height under upper nodes.
    threshold = 0.5 * (1 + landing) * g[0]
    ref = rest_shape(g)
    bottom = tri[(ref[tri, 0] < threshold + 0.01 * g[0]).all(-1)]
    top = last[ref[:, 0] > threshold + 0.12 * g[0]]
    heights = []
    for point in top:
        p = last[bottom]
        s1 = p[:, 1, :2] - p[:, 0, :2]
        s2 = p[:, 2, :2] - p[:, 0, :2]
        dxy = point[:2] - p[:, 0, :2]
        cross2 = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        det = cross2(s1, s2)
        safe = np.where(abs(det) > 1e-12, det, 1e-12)
        u = cross2(dxy, s2) / safe
        v = cross2(s1, dxy) / safe
        valid = (
            (u >= -0.00001) & (v >= -0.00001) & (u + v <= 1.00001) & (abs(det) > 1e-12)
        )
        if np.any(valid):
            z = (1 - u - v) * p[:, 0, 2] + u * p[:, 1, 2] + v * p[:, 2, 2]
            heights.append(point[2] - np.max(z[valid]))
    return {
        "strict_edge_face_crossings": int(crossed.sum()),
        "min_z_m": float(last[:, 2].min()),
        "overlap_probe_count": len(heights),
        "positive_layer_fraction": float(np.mean(np.array(heights) > 0.001))
        if heights
        else None,
        "median_layer_gap_mm": float(1000 * np.median(heights)) if heights else None,
    }


def get_path(folder, i, j):
    return folder / f"test-{i:02}-{j:02}.npz"


@torch.no_grad()
def plan(model, g, actions, landing):
    gs = np.broadcast_to(g, (len(actions), len(g))).copy()
    predicted = model(torch.from_numpy(gs), torch.from_numpy(actions)).numpy()
    ms = [metrics(q, g, landing) for q in predicted]
    best = min(range(len(ms)), key=lambda j: score(ms[j], g))
    return best, predicted, ms


def evaluate(root):
    torch.set_num_threads(1)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        fit(root / "train", root / "fold-response.npz")
        times.append(time.perf_counter() - start)
    model = FoldResponse(root / "fold-response.npz").eval()
    report = []
    for i in range(TEST):
        seed, landing, act = task(i)
        g = geometry(seed)
        selected, predictions, pred_metrics = plan(model, g, act, landing)
        moves = []
        shapes = []
        teacher_diagnostics = []
        for j in range(len(act)):
            with np.load(get_path(root / "test", i, j)) as z:
                assert int(z["seed"]) == seed and np.array_equal(z["a"], act[j])
                q = z["q"]
                moves.append(metrics(q, g, landing))
                shapes.append(q)
                teacher_diagnostics.append(
                    {
                        k: float(z[k])
                        for k in ("max_edge_stretch", "min_z_m", "max_table_candidates")
                    }
                )
        oracle = min(range(len(act)), key=lambda j: score(moves[j], g))
        ids = {"no_move": 0, "script": 1, "model": selected, "newton": oracle}
        audits = {
            name: (contact_audit(shapes[j], g, landing) | teacher_diagnostics[j])
            for name, j in ids.items()
        }
        chosen = shapes[selected]
        pred = predictions[selected]
        rec = {
            "seed": seed,
            "length_m": float(g[0]),
            "landing_fraction": landing,
            "indices": ids,
            "native": {name: moves[j] for name, j in ids.items()},
            "native_contact": audits,
            "predicted": pred_metrics[selected],
            "predicted_contact": contact_audit(pred, g, landing),
            "selected_rollout_error_percent": float(
                100
                * np.linalg.norm(pred[1:] - chosen[1:], axis=-1)
                .__pow__(2)
                .mean(axis=1)
                .__pow__(0.5)
                .mean()
                / g[0]
            ),
        }
        if i == 0:
            np.savez_compressed(
                root / "example.npz",
                teacher=chosen,
                predicted=pred,
                script=shapes[1],
                goal=target(chosen[0], g[0], landing),
                material=g,
                action=act[selected],
                time=DT * np.arange(len(chosen)),
                seed=seed,
            )
        report.append(rec)
    summary = {}
    for kind in ("no_move", "script", "model", "newton"):
        rows = [r["native"][kind] for r in report]
        summary[kind] = {
            k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "success"
        }
        summary[kind]["success_count"] = sum(r["success"] for r in rows)
        audits = [r["native_contact"][kind] for r in report]
        summary[kind]["terminal_intersecting_cases"] = sum(
            a["strict_edge_face_crossings"] > 0 for a in audits
        )
        summary[kind]["median_layer_gap_mm"] = (
            float(
                np.median(
                    [
                        a["median_layer_gap_mm"]
                        for a in audits
                        if a["median_layer_gap_mm"] is not None
                    ]
                )
            )
            if any(a["median_layer_gap_mm"] is not None for a in audits)
            else None
        )
    summary["model"]["predicted_success_count"] = sum(
        r["predicted"]["success"] for r in report
    )
    summary["model"]["predicted_native_success_disagreement"] = sum(
        r["predicted"]["success"] != r["native"]["model"]["success"] for r in report
    )
    summary["model"]["mean_rollout_error_pct"] = float(
        np.mean([r["selected_rollout_error_percent"] for r in report])
    )
    result = {
        "summary": summary,
        "cases": report,
        "fit_wall_s_data_read_and_checkpoint_write": times,
        "train_capture": json.loads((root / "train/capture.json").read_text()),
        "test_capture": json.loads((root / "test/capture.json").read_text()),
    }
    (root / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.root)


if __name__ == "__main__":
    main()
