"""Audit compact cable and cloth rollouts on Newton's native vertices and paired forces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pack_deformables import interpolation


def pack(source: Path, output: Path) -> None:
    manifest = json.loads((source / "manifest.json").read_text())
    native = {}
    for record in manifest["episodes"]:
        key = record["identity"]
        with np.load(source / record["file"]) as data:
            native[key] = data["position"]
            native[key + "_rest"] = data["rest_position"]
            native[key + "_free"] = data["inverse_mass"].reshape(-1) > 0
    np.savez_compressed(output, **native)


def decode(displacement, static, rest, free):
    cloth, length, _, nx, ny = static[:5]
    axial = interpolation(5, int(nx) + 1)
    lateral = interpolation(3, int(ny) + 1) if cloth else np.full((1, 3), 1 / 3)
    decoder = np.kron(lateral, axial).astype(np.float32)
    result = rest + length * (decoder @ displacement)
    result[:, ~free] = rest[~free]
    return result


def frame_error(delta):
    return np.sqrt(np.mean(np.sum(delta.astype(np.float64)**2, axis=-1), axis=-1))


def evaluate(packed, native, predictions, start, *, native_positions=False):
    q, u = packed["q"], packed["u"]
    static = packed["static"]
    names = packed["identity"].tolist()
    geometry = packed["seed"]
    material = packed["family"]
    if not native_positions and (predictions.shape != q[:, start + 1:].shape or not np.isfinite(predictions).all()):
        raise ValueError("Expected a finite prediction for each future step, trajectory and canonical point.")
    cases, pairs = [], []
    decoded = []
    for i, key in enumerate(names):
        rest = native[key + "_rest"]
        free = native[key + "_free"]
        actual = native[key][start + 1:]
        if native_positions:
            estimated = predictions[key][start + 1:]
            if estimated.shape != actual.shape or not np.isfinite(estimated).all():
                raise ValueError(f"Invalid native world positions for {key}")
        else:
            estimated = decode(predictions[i], static[i], rest, free)
        interpolated = decode(q[i, start + 1:], static[i], rest, free)
        decoded.append(estimated)
        err = frame_error(estimated - actual)
        cases.append({
            "identity": key, "family": material[i], "seed": int(geometry[i]),
            "native_mm": float(1000 * err.mean()), "native_pct": float(100 * err.mean() / static[i, 1]),
            "interpolation_mm": float(1000 * frame_error(interpolated - actual).mean()),
            "hold_mm": float(1000 * frame_error(native[key][start] - actual).mean()),
            "terminal_mm": float(1000 * err[-1]),
        })
    for i, key in enumerate(names):
        if packed["excitation"][i] == "reference":
            continue
        ref, = np.flatnonzero((geometry == geometry[i]) & (material == material[i]) & (packed["excitation"] == "reference"))
        changes = np.flatnonzero(np.linalg.norm(u[i] - u[ref], axis=1) > 1e-5)
        first = max(start, int(changes[0]))
        base = names[ref]
        target = native[key][first+1:] - native[base][first+1:]
        obtained = (decoded[i] - decoded[ref])[first-start:]
        magnitude = np.sqrt(np.mean(frame_error(target)**2))
        residual = np.sqrt(np.mean(frame_error(obtained-target)**2))
        pairs.append({
            "identity": key, "family": material[i], "seed": int(geometry[i]),
            "response_mm": float(1000*magnitude), "error_mm": float(1000*residual),
            "response_error_pct": float(100*residual/magnitude),
        })
    summary = {}
    for family in ("cable", "cloth"):
        rows = [c for c in cases if c["family"] == family]
        active = [c for c in pairs if c["family"] == family]
        summary[family] = {field: float(np.mean([v[field] for v in rows]))
                           for field in ("native_mm", "native_pct", "interpolation_mm", "hold_mm", "terminal_mm")}
        summary[family]["response_error_pct"] = float(np.mean([v["response_error_pct"] for v in active]))
        summary[family]["force_pairs_improving_over_zero"] = sum(v["response_error_pct"] < 100 for v in active)
        summary[family]["force_pairs"] = len(active)
    return {"summary": summary, "cases": cases, "force_pairs": pairs}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    packing = sub.add_parser("pack")
    packing.add_argument("source", type=Path)
    packing.add_argument("output", type=Path)
    scoring = sub.add_parser("score")
    scoring.add_argument("data", type=Path)
    scoring.add_argument("native", type=Path)
    scoring.add_argument("predictions", type=Path)
    scoring.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.command == "pack":
        pack(args.source, args.output)
        return
    with np.load(args.data) as dataset, np.load(args.native) as archive, np.load(args.predictions) as preds:
        packed, native = dict(dataset), dict(archive)
        if "identity" in preds and not np.array_equal(packed["identity"], preds["identity"]):
            raise ValueError("Mismatched prediction identities")
        report = {}
        for start in (0, 100):
            if f"q_from_{start}" in preds:
                trajectory = preds[f"q_from_{start}"]
            else:
                trajectory = preds[f"q{start}"]
                if not np.allclose(trajectory[:, 0], packed["q"][:, start], atol=1e-5):
                    raise ValueError("First frame disagrees with the observed restart")
                trajectory = trajectory[:, 1:]
            report[str(start)] = evaluate(packed, native, trajectory, start)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
