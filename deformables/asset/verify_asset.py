"""Independently refit the released calibration data and compare four models on new programs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import asset_response as first
from . import calibrate_response as second


def verify(args):
    exploratory = json.loads(args.exploratory.read_text())
    released = json.loads(args.reference.read_text())
    output = {"assets": [], "methods": {}, "max_replay_error_mm": 0.0}
    for item in exploratory["assets"]:
        family, seed = item["family"], item["seed"]
        key = f"{family}-{seed}"
        calibration = args.calibration / f"{key}.npz"
        fresh = args.fresh / f"fresh-{key}.npz"
        digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
        if digest != item["data_sha256"]:
            raise ValueError(f"Calibration checksum mismatch for {key}")
        with np.load(calibration) as train, np.load(fresh) as test:
            nominal = train["nominal"][None, 1:].astype(np.float64)
            y = train["train_q"].astype(np.float64) - nominal
            u = train["train_u"]
            new = test["u"]
            target = test["y"].astype(np.float64)
            for a in new:
                if any(np.array_equal(a, b) for b in np.concatenate([u, train["test_u"]])):
                    raise ValueError(f"Duplicate fresh program in {key}")
            np.testing.assert_array_equal(test["repeat"], train["nominal"])
            np.testing.assert_array_equal(test["repeat0"], train["test_q"][0])
            model = second.fit(u, y)
            predicted = second.predict(model, new, nominal)
            changed = new.copy()
            changed[:, 40:] *= -1.5
            np.testing.assert_allclose(second.predict(model, changed, nominal)[:, :40],
                                       predicted[:, :40], rtol=0, atol=1e-12)
            np.testing.assert_allclose(second.predict(model, np.zeros_like(new), nominal),
                                       np.broadcast_to(nominal, target.shape), rtol=0, atol=1e-12)
            predictions = {"rbf": predicted, "nominal": nominal}
            for method in first.METHODS:
                beta = first.fit(u, y, method)
                correction = first.features(new, method) @ beta
                predictions[method] = nominal + correction.reshape(target.shape)
            rows = {name: first.metrics(q, target, nominal, item["asset_span_m"])
                    for name, q in predictions.items()}
            reference = next(r for r in released["assets"] if r["seed"] == seed)
            for actual, saved in zip(rows["rbf"], reference["fit"]["12"]["test"], strict=True):
                delta = abs(actual["rms_mm"] - saved["rms_mm"])
                if delta > 1e-6:
                    raise ValueError(f"Score differs for {key} by {delta} mm")
                output["max_replay_error_mm"] = max(delta, output["max_replay_error_mm"])
            output["assets"].append({"family": family, "seed": seed, "nodes": item["nodes"],
                                     "calibration_sha256": digest,
                                     "fresh_sha256": hashlib.sha256(fresh.read_bytes()).hexdigest(), "rows": rows})
    for family in ("cable", "cloth"):
        assets = [a for a in output["assets"] if a["family"] == family]
        output["methods"][family] = {}
        for method in ("nominal", *first.METHODS, "rbf"):
            rows = [row for a in assets for row in a["rows"][method]]
            output["methods"][family][method] = {
                **first.summarize(rows),
                "cases_improved": sum(r["percent_response"] < 100 - 1e-9 for r in rows),
                "cases": len(rows),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["methods"], indent=2))


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=root / "data/calibration")
    parser.add_argument("--fresh", type=Path, default=root / "data/fresh")
    parser.add_argument("--exploratory", type=Path, default=root / "results/exploratory.json")
    parser.add_argument("--reference", type=Path, default=root / "results/fresh.json")
    parser.add_argument("--output", type=Path, default=root / "results/verified.json")
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
