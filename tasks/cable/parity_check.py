"""Check the deployed NeurALL portable adapter against the predecessor's packed CPU estimator."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cable_gates import (
    COMMANDS,
    FRAMES,
    SEEDS,
    PortableRecurrent,
    Teacher,
    neurall_predict,
    pack,
)
from experiments.neurall.elastic_recurrent import PackedElastic, RecurrentElastic

p = argparse.ArgumentParser()
p.add_argument("--original", type=Path, required=True)
p.add_argument("--adapter", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
torch.set_num_threads(1)
orig = torch.load(args.original, map_location="cpu", weights_only=True)
adapter = torch.load(args.adapter, map_location="cpu", weights_only=True)
net = RecurrentElastic(len(orig["center"]), orig["width"]).eval()
net.load_state_dict(orig["model"])
portable = PortableRecurrent(adapter).eval()
scene = Teacher(SEEDS[0])
ep = scene.episode
meta = dict(
    ep["metadata"],
    family="cable",
    seed=SEEDS[0],
    identity="parity",
    excitation="ordinary",
    sha256="",
)
sample = pack(meta, ep)
forces = np.broadcast_to(sample["u"], (25, FRAMES - 1, 3)).copy()
forces[:, :, 1] += COMMANDS @ scene.shapes[:-1].T
with torch.inference_mode():
    pred = neurall_predict(adapter, portable, sample, ep, forces)
deltas = []
for i, seq in enumerate(forces):
    packed = PackedElastic(
        net,
        orig["center"].numpy(),
        orig["scale"].numpy(),
        orig["bases"]["cable"].numpy(),
        ep,
        meta,
        project_cable_length=orig["project_cable_length"],
    )
    other = np.stack([packed.step(c) for c in seq])
    deltas.append(float(np.max(np.abs(other - pred[i]))))
assert max(deltas) < 2e-5
args.output.write_text(
    json.dumps(
        {
            "metric": "maximum component difference across all 25 x 120 native rollouts, m",
            "seed": SEEDS[0],
            "maximum_m": max(deltas),
            "per_program_m": deltas,
        },
        indent=2,
    )
    + "\n"
)
print(max(deltas))
