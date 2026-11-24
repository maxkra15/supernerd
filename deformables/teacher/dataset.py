# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare and load versioned NeurALL benchmarks in world SI units."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MATERIAL_IDS = {"sand": 0, "cloth": 1, "cable": 2}
SCHEMA = 1


def digest(path):
    """Return a file's SHA-256 fingerprint."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _material_parameters(episode, family):
    meta = episode["metadata"]
    if family == "sand":
        return (meta["friction"], 0, 0)
    if family == "cloth":
        stiffness = meta.get("membrane_stiffness", meta.get("cloth_membrane_stiffness", 100))
        return (np.log10(1 + stiffness) / 6, min(meta.get("membrane_damping", 0), 10) / 10, 0)
    return (
        np.log10(1 + meta.get("stretch_stiffness_n_m", meta.get("cable_stretch_stiffness_n_m", 100))) / 6,
        np.log10(1 + meta.get("bend_stiffness_nm_rad", meta.get("cable_bend_stiffness_nm_rad", 0))) / 6,
        meta.get("radius_m", meta.get("cable_radius_m", 0)) / 0.05,
    )


def _validate(arrays, metadata):
    q, time = arrays["position"], arrays["time"]
    if q.ndim != 3 or q.shape[-1] != 3 or len(time) != len(q) or len(time) < 4:
        raise ValueError(f"Invalid state shape in {metadata['identity']}")
    count = q.shape[1]
    if not np.allclose(np.diff(time), metadata["dt"], atol=7e-7, rtol=0):
        raise ValueError(f"Nonuniform clock in {metadata['identity']}")
    shapes = {
        "rest_position": (count, 3),
        "material_id": (count,),
        "material_parameters": (count, 3),
        "inverse_mass": (count,),
        "gravity": (3,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name}: {arrays[name].shape} differs from {shape}")
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError(f"Nonfinite arrays in {metadata['identity']}")
    edges = arrays["edge_index"]
    if edges.ndim != 2 or edges.shape[0] != 2 or (edges.size and (edges.min() < 0 or edges.max() >= count)):
        raise ValueError("Invalid topology indices")
    if arrays["rest_length"].shape != (edges.shape[1],) or arrays["edge_kind"].shape != (edges.shape[1],):
        raise ValueError("Topology labels and rest lengths must align")


def _write_episode(output, identity, family, split, episode, provenance):
    metadata = episode["metadata"] | {
        "identity": identity,
        "family": family,
        "split": split,
        "provenance": provenance,
    }
    position = episode["position"]
    arrays = {name: np.asarray(value) for name, value in episode.items() if name != "metadata"}
    count = position.shape[1]
    if "material_id" not in arrays:
        arrays["material_id"] = np.full(count, MATERIAL_IDS[family], dtype=np.int64)
    if "material_parameters" not in arrays:
        params = np.empty((count, 3), dtype=np.float32)
        for name, value in MATERIAL_IDS.items():
            selected = arrays["material_id"] == value
            if selected.any():
                params[selected] = _material_parameters(episode, name)
        arrays["material_parameters"] = params
    _validate(arrays, metadata)
    path = output / f"{identity}.npz"
    np.savez_compressed(path, **arrays)
    return {
        **metadata,
        "file": path.name,
        "sha256": digest(path),
        "particles": count,
        "states": len(position),
    }


def _sand_episode(scene, data_root):
    import h5py  # noqa: PLC0415 -- only the public corpus adapter uses HDF5

    q = scene["position"]
    meta = scene["metadata"]
    native = Path(data_root) / scene["identity"]["path"]
    with h5py.File(native, "r") as stream:
        wrench = np.asarray(stream["body/contact_wrench"][: (len(q) - 1) * 4], dtype=np.float32)
        body_mass = float(stream.attrs.get("object_mass_kg", 2.0))
    # Translational impulse in world frame. Torque is retained at the
    # original cadence. Its moment reference changes with the body COM.
    impulse = (wrench[:, :3].reshape(-1, 4, 3) * 0.005).sum(1)
    return {
        "time": scene["time"],
        "position": q,
        "rest_position": q[0],
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "edge_kind": np.empty(0, dtype=np.int64),
        "rest_length": np.empty(0, dtype=np.float32),
        "inverse_mass": 1 / meta["particle_mass"],
        "gravity": np.asarray(meta["gravity"], dtype=np.float32),
        "tool_pose": scene["collider_pose"],
        "tool_twist": scene["collider_velocity"],
        "tool_half_extents": meta["collider_half_extents"],
        "tool_native_wrench": wrench,
        "tool_impulse": impulse,
        "tray_pose": meta["static_collider_pose"],
        "tray_half_extents": meta["static_collider_half_extents"],
        "metadata": {
            "dt": meta["prediction_interval"],
            "tool_type": meta["collider_type"],
            "tool_mass_kg": body_mass,
            "friction": float(meta["material_native"]["friction"]),
            "velocity_convention": "Position secants at 50 Hz; initial velocity zero",
            "prediction_condition": "Recorded body track supplied",
        },
    }


def prepare(data, output, native_train, native_test, frames, native_only):
    """Export complete, identity-disjoint native and public episodes."""
    from .teachers import generate_episode  # noqa: PLC0415 -- optional native dependencies

    output.mkdir(parents=True, exist_ok=False)
    records = []
    for family in ("cloth", "cable", "mixed"):
        count = max(2, native_train // 2) if family == "mixed" else native_train
        for split, start, size in (("train", 1000, count), ("test", 9000, native_test)):
            for index in range(size):
                seed = start + index
                episode = generate_episode(family, seed, frames=frames, dt=0.02)
                records.append(_write_episode(output, f"{family}-{seed}", family, split, episode, "Newton native"))
                print(f"{family}-{seed} {split} {records[-1]['particles']} particles", flush=True)
    if not native_only:
        from ..mpm_surface.public_benchmark.data import PublicBenchmarkDataset  # noqa: PLC0415
        from ..mpm_surface.public_benchmark.replay_impact import load_impact_replay, load_rest_replay  # noqa: PLC0415

        groups = (
            ("train", PublicBenchmarkDataset(data, "train", verify_hashes=True)),
            ("train", load_impact_replay(data, indices=(0, 3))),
            ("train", load_rest_replay(data, indices=(0,))),
            ("diagnostic", PublicBenchmarkDataset(data, "test", verify_hashes=True, allow_reserved=True)),
            ("diagnostic", load_rest_replay(data, indices=(1,))),
        )
        for split, group in groups:
            for source in group:
                identity = source["identity"]["trajectory"]
                episode = _sand_episode(source, data)
                episode["metadata"]["source"] = source["identity"]
                episode["metadata"]["source_sha256"] = source["source_sha256"]
                records.append(_write_episode(output, identity, "sand", split, episode, "Pinned public Newton MPM"))
                print(f"{identity} {split} {records[-1]['particles']} particles", flush=True)
    manifest = {"schema": SCHEMA, "episodes": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(records)} complete episodes to {output}", flush=True)


def load_manifest(root):
    """Load a complete manifest and check that trajectory identities are unique."""
    value = json.loads((Path(root) / "manifest.json").read_text())
    records = value["episodes"]
    if value["schema"] != SCHEMA or len(records) != len({row["identity"] for row in records}):
        raise ValueError("Unsupported manifest or a duplicate trajectory identity")
    return records


def load_episode(root, record):
    """Load a pinned episode. Derive interval velocities when only positions were logged."""
    path = Path(root) / record["file"]
    if path.name != record["file"] or digest(path) != record["sha256"]:
        raise ValueError(f"Dataset fingerprint differs: {record['identity']}")
    with np.load(path, allow_pickle=False) as stream:
        arrays = {name: stream[name] for name in stream.files}
    if "velocity" not in arrays:
        q = arrays["position"]
        velocity = np.zeros_like(q)
        velocity[1:] = (q[1:] - q[:-1]) / record["dt"]
        arrays["velocity"] = velocity
    if "external_acceleration" not in arrays:
        arrays["external_acceleration"] = np.zeros((len(arrays["time"]), 1, 3), dtype=np.float32)
    return arrays


def main():
    """Write the first benchmark version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/mpm-public-8942"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-train", type=int, default=24)
    parser.add_argument("--native-test", type=int, default=4)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    if min(args.native_train, args.native_test) < 1 or args.frames < 4:
        parser.error("Require separate nonempty split sizes and at least four frames")
    prepare(args.data, args.output, args.native_train, args.native_test, args.frames, args.native_only)


if __name__ == "__main__":
    main()
