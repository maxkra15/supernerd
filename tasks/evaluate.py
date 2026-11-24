"""Score a planner's discrete action choices against the published Newton executions."""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def cable_cases():
    folder = ROOT / "cable/results"
    bank = read(folder / "candidate-bank.json")
    cases = {}
    for seed, motions in bank.items():
        commands = motions["commands"]
        tips = motions["gate_y_m"]
        if len(commands) != 25 or len(tips) != len(commands):
            raise ValueError(f"Incomplete cable candidate bank for {seed}")
        for target in read(folder / f"object-{seed}.json")["cases"]:
            wanted = target["gate_y_m"]
            results = []
            for tip in tips:
                worst = max(abs(y - goal) for y, goal in zip(tip, wanted, strict=True))
                results.append(
                    {"passed": worst <= 0.010, "worst_gate_error_mm": 1000 * worst}
                )
            identity = f"{seed}/{target['target']}"
            cases[identity] = (
                {"seed": int(seed), "gate_y_m": wanted, "commands": commands},
                results,
            )
    return cases


def cloth_cases():
    cases = {}
    for fabric in read(ROOT / "cloth/all-candidates.json")["confirm"]:
        methods = fabric["candidates"]
        if [x["index"] for x in methods] != list(range(12)):
            raise ValueError(f"Incomplete cloth candidate bank for {fabric['seed']}")
        description = {key: fabric[key] for key in ("geometry", "landing", "seed")}
        description["commands"] = [x["action"] for x in methods]
        outcomes = [
            {
                "passed": x["teacher"]["success"],
                "shape_error_pct": x["teacher"]["shape_error_percent"],
            }
            for x in methods
        ]
        cases[str(fabric["seed"])] = (description, outcomes)
    return cases


def read_choices(path):
    def unique_keys(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(f"Duplicate submission key {key!r}")
            data[key] = value
        return data

    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


def score(cases, choices):
    if not isinstance(choices, dict) or set(choices) != set(cases):
        raise ValueError("Supply exactly one choice for every listed case")
    selected = {}
    for case, (_, outcomes) in cases.items():
        index = choices[case]
        if type(index) is not int or not 0 <= index < len(outcomes):
            raise ValueError(
                f"Invalid zero-based candidate index {index!r} for case {case}"
            )
        selected[case] = {"candidate": index, **outcomes[index]}
    numeric_key = next(key for key in outcomes[0] if key != "passed")
    return {
        "passes": sum(row["passed"] for row in selected.values()),
        "cases": len(selected),
        f"mean_{numeric_key}": math.fsum(row[numeric_key] for row in selected.values())
        / len(selected),
        "executions": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["cable", "cloth"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="Print task inputs, goals and indexed commands",
    )
    group.add_argument(
        "--score",
        type=Path,
        metavar="CHOICES.json",
        help="JSON object mapping case IDs to candidate indices",
    )
    args = parser.parse_args()
    cases = cable_cases() if args.task == "cable" else cloth_cases()
    if args.list:
        data = {identity: info for identity, (info, _) in cases.items()}
    else:
        data = score(cases, read_choices(args.score))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
