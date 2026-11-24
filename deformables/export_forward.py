"""Export ordered autonomous trajectories for independent native-node evaluation."""

import argparse
from pathlib import Path

import numpy as np
import torch

from modal_forward import ModalForward, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/test.npz"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high" if args.device == "cuda" else "highest")
    model = ModalForward().to(args.device).eval()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    data = load(args.data, args.device)
    predictions = {}
    with torch.no_grad():
        for t in (0, 100):
            predictions[f"q_from_{t}"] = model(
                data["q"][:, t], data["v"][:, t], data["static"], data["u"][:, t:]
            ).cpu().numpy()
    np.savez_compressed(args.output, identity=data["identity"], **predictions)


if __name__ == "__main__":
    main()
