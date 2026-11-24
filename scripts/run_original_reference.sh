#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo 'Usage: run_original_reference.sh <NeRD main checkout> <output directory>' >&2
    exit 2
fi

checkout=$(cd "$1" && pwd)
mkdir -p "$2"
output=$(cd "$2" && pwd)
checkpoint="$checkout/pretrained_models/NeRD_models/Cartpole/model/nn/model.pt"
if [[ "$(git -C "$checkout" rev-parse HEAD)" != 370e89d2b8ce68aa3578aa7812859c681b7502bc ]]; then
    echo 'Use the pinned original NeRD checkout 370e89d2b8ce68aa3578aa7812859c681b7502bc.' >&2
    exit 2
fi
echo "287dde2134eb8f5277c17a8f7cc7abbe64063e97cb481bdb6e8d76f9536e5366  $checkpoint" | sha256sum -c -

cd "$checkout/eval/eval_passive"
for horizon in 100 500 1000; do
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 uv run --no-project python eval_passive_motion.py \
        --env-name Cartpole --model-path "$checkpoint" --env-mode neural \
        --num-envs 2048 --num-rollouts 2048 --rollout-horizon "$horizon" --seed 500 \
        >"$output/h${horizon}.log" 2>&1
    awk '/Base position error mean|Joint position error mean/ { print }' "$output/h${horizon}.log"
done

uv run --no-project python - "$output" <<'PY'
import json
import re
import sys
from pathlib import Path

import torch
import warp

output = Path(sys.argv[1])
result = {
    "original_nerd_commit": "370e89d2b8ce68aa3578aa7812859c681b7502bc",
    "checkpoint_sha256": "287dde2134eb8f5277c17a8f7cc7abbe64063e97cb481bdb6e8d76f9536e5366",
    "python": sys.version.split()[0],
    "warp": warp.__version__,
    "torch": torch.__version__,
    "device": torch.cuda.get_device_name(),
    "seed": 500,
    "cases": 2048,
    "mean_absolute_error_over_time_and_cases": {},
}
for horizon in (100, 500, 1000):
    log = (output / f"h{horizon}.log").read_text()
    cart = re.search(r"Base position error mean\s+= ([\d.]+)", log)
    pole = re.search(r"Joint position error mean\s+= ([\d.]+)", log)
    result["mean_absolute_error_over_time_and_cases"][str(horizon)] = {
        "cart_m": float(cart.group(1)),
        "pole_rad": float(pole.group(1)),
    }
(output / "replication.json").write_text(json.dumps(result, indent=2) + "\n")
PY
