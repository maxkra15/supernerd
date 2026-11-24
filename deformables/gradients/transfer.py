"""Check fixed adaptations against the previously published, separate chirp/triangle bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gradients.local_fit import initial_model
from native_rollouts import evaluate


@torch.no_grad()
def score(model, packed, native):
    output = {}
    for start in (0, 100):
        values = (
            packed["q"][:, start],
            packed["v"][:, start],
            packed["static"],
            packed["u"][:, start:],
        )
        predicted = model(*(torch.from_numpy(v) for v in values)).numpy()
        output[str(start)] = evaluate(packed, native, predicted, start)["summary"]
    return output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--models", nargs="+", type=Path, required=True)
    cli.add_argument("--data", default=Path("data"), type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    with (
        np.load(args.data / "fresh.npz") as f,
        np.load(args.data / "native-fresh.npz") as n,
    ):
        packed, native = dict(f), dict(n)
    result = {}
    for model in args.models:
        result[model.stem] = score(initial_model(model, "cpu").eval(), packed, native)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
