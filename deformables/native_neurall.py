"""Audit NeurALL's archived frame-zero rollouts on the fixed native cable/cloth test.

To reproduce the published score from the deformables directory:
    python native_neurall.py score data results/neurall-native-positions.npz results/native-neurall.json

To rebuild the predictions archive from NeurALL's load-diversity-v5-eval-held-diverse:
    python native_neurall.py pack data ORIGINAL_RUN_DIRECTORY results/neurall-native-positions.npz
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from native_rollouts import evaluate


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pack(data, native, source, output):
    names = data["identity"].tolist()
    metrics = {r["identity"]: r for r in json.loads((source / "metrics.json").read_text())}
    if set(names) != {path.stem for path in source.glob("*.npz")} or set(metrics) != set(names):
        raise ValueError("Trajectory identity mismatch")
    expected_times = np.arange(data["q"].shape[1], dtype=np.float64) * 0.02
    result = {"identity": data["identity"], "time": expected_times}
    hashes, lengths = [], []
    for i, key in enumerate(names):
        with np.load(source / (key + ".npz")) as saved:
            positions, teacher = saved["neurall"], saved["teacher"]
            if teacher.dtype != native[key].dtype or not np.array_equal(teacher, native[key]):
                raise ValueError(f"Archived teacher differs from the native benchmark for {key}")
            if positions.shape != teacher.shape or not np.array_equal(positions[0], teacher[0]):
                raise ValueError(f"Array shape or initial position mismatch for {key}")
            if not np.allclose(saved["time"], expected_times, rtol=0, atol=1e-12):
                raise ValueError(f"Time grid mismatch for {key}")
            if metrics[key]["family"] != data["family"][i]:
                raise ValueError(f"Material family mismatch for {key}")
            result[key] = positions
            hashes.append(digest(teacher))
            lengths.append(metrics[key]["length_m"])
    result["teacher_sha256"] = np.array(hashes)
    result["length"] = np.array(lengths)
    np.savez_compressed(output, **result)


def score(data, native, source, output):
    with np.load(source) as archive:
        if not np.array_equal(archive["identity"], data["identity"]):
            raise ValueError("Archive identity/order mismatch")
        if not np.allclose(archive["length"], data["length"], rtol=0, atol=1e-7):
            raise ValueError("Rest-length mismatch")
        positions = {}
        for i, key in enumerate(data["identity"]):
            positions[key] = archive[key]
            if digest(native[key]) != archive["teacher_sha256"][i]:
                raise ValueError(f"Native teacher hash mismatch for {key}")
            if not np.array_equal(positions[key][0], native[key][0]):
                raise ValueError(f"Frame-zero mismatch for {key}")
        if not np.allclose(archive["time"], np.arange(data["q"].shape[1]) * 0.02, rtol=0, atol=1e-12):
            raise ValueError("Archived time grid mismatch")
        result = {"0": evaluate(data, native, positions, 0, native_positions=True)}
        result["verification"] = {
            "matched_trajectories": len(positions),
            "frames_per_trajectory": len(archive["time"]),
            "frame_interval_seconds": 0.02,
            "native_teacher_hashes_matched": len(positions),
            "max_rest_length_difference_m": float(abs(archive["length"] - data["length"]).max()),
            "source": "NeurALL, load-diversity-v5-eval-held-diverse",
        }
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["score", "pack"])
    parser.add_argument("data", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.data / "test.npz") as test, np.load(args.data / "native-test.npz") as native:
        task = pack if args.command == "pack" else score
        task(dict(test), dict(native), args.source, args.output)


if __name__ == "__main__":
    main()
