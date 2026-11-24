"""Repack public-safe Newton trajectory fields onto a shared observation grid."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

SOURCES = {
    "train": "load-diversity-v5-diverse",
    "development": "load-diversity-v5-development",
    "test": "load-diversity-v5-held",
}
FEATURES = [
    "is_cloth",
    "length_m",
    "width_m",
    "axial_cells",
    "lateral_cells",
    "mass_per_free_vertex_kg",
    "log_stiffness",
    "membrane_damping",
    "bend_stiffness_nm_rad",
    "radius_m",
    "density_kg_m3",
    "initial_tilt_rad",
]


def interpolation(source, target):
    grid = np.linspace(0, source - 1, target)
    lo = np.floor(grid).astype(int)
    frac = grid - lo
    return (
        np.eye(source)[lo] * (1 - frac[:, None])
        + np.eye(source)[np.minimum(lo + 1, source - 1)] * frac[:, None]
    )


def pack(record, data):
    cloth = record["family"] == "cloth"
    nx, ny = tuple(record["cells"]) if cloth else (record["segment_count"], 0)
    mapper = np.kron(interpolation(ny + 1, 3), interpolation(nx + 1, 5)).astype("f")
    length = record["length_m"]
    rest = mapper @ data["rest_position"]
    q = (mapper @ data["position"] - rest) / length
    v = mapper @ data["velocity"] / length
    statics = np.array(
        [
            cloth,
            length,
            record.get("width_m", 0),
            nx,
            ny,
            record.get("mass_per_free_vertex_kg", 0),
            np.log(
                record.get("membrane_stiffness", record.get("stretch_stiffness_n_m"))
            ),
            record.get("membrane_damping", 0),
            record.get("bend_stiffness_nm_rad", 0),
            record.get("radius_m", 0),
            record.get("density_kg_m3", 0),
            record["initial_tilt_rad"],
        ],
        dtype="f",
    )
    return {
        "q": q,
        "v": v,
        "u": data["external_acceleration"][:-1, -1],
        "rest": rest,
        "static": statics,
        "length": length,
        "family": record["family"],
        "identity": record["identity"],
        "seed": record["seed"],
        "excitation": record["excitation"],
        "source_sha256": record["sha256"],
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    identities = set()
    receipt = {
        "features": FEATURES,
        "sampling": "3 lateral rows then 5 axial points, bilinear interpolation",
        "dt_s": 0.02,
        "source_dirs": SOURCES,
        "splits": {},
    }
    for split, name in SOURCES.items():
        start = perf_counter()
        folder = args.source / name
        episodes = json.loads((folder / "manifest.json").read_text())["episodes"]
        episodes = sorted(
            episodes, key=lambda r: (r["family"], r["seed"], r["identity"])
        )
        keys = {(r["family"], r["seed"]) for r in episodes}
        assert keys.isdisjoint(identities)
        identities |= keys
        samples = []
        for record in episodes:
            assert (
                record["split"] == split
                and record["states"] == 301
                and record["dt"] == 0.02
            )
            path = folder / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Trajectory digest differs for {record['identity']}")
            with np.load(path) as archive:
                samples.append(pack(record, archive))
        data = {k: np.stack([sample[k] for sample in samples]) for k in samples[0]}
        output = args.output / f"{split}.npz"
        np.savez_compressed(output, **data)
        receipt["splits"][split] = {
            "episodes": len(episodes),
            "objects": len(keys),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "pack_s": perf_counter() - start,
        }
        print(split, receipt["splits"][split], flush=True)
    (args.output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
