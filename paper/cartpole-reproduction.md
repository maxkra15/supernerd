# Cartpole experiment

The pilot uses the public NeRD Cartpole body and action conventions. Newton Featherstone advances five 1/300 s substeps at each policy frame. Policy actions are limited to ±225 N to study local balance. The model takes the current cart position, pole angle, both velocities, and normalized force. The original NeRD checkpoint uses a 10-frame history and maximal-coordinate inputs. The experiment evaluates short-horizon policy learning and long-horizon forward dynamics at separate timescales.

## Repeat the matched pilot

Run from the SuperNeRD source root. The official asset lives in the public NeRD repository (Apache-2.0). Fetch it from the immutable commit and verify its contents before running.

```sh
curl -fL https://raw.githubusercontent.com/NVlabs/neural-robot-dynamics/476dc67fccc6a99be293bf611f897b1c06f13e30/envs/newton_envs/assets/cartpole_single.urdf -o cartpole_single.urdf
sha256sum cartpole_single.urdf
```

Expected SHA-256 is `f31276531c03050a3ac3a3e24443899f38a4424586db17140978383433dec237`. The benchmark also checks it before creating any worlds.

The measured run used Newton source revision `0ced8f8420d4ee1eab9ecfaee59e1354ef670348`, Warp `1.19.0.dev20261020`, NumPy `2.5.3`, PyTorch `2.15.0+cu132`, Python `3.12.14` and an RTX PRO 6000 Blackwell. Install the pinned Newton source in a uv environment with the matching Warp and PyTorch wheel builds. Newton's online package indexes are listed in its `pyproject.toml`; PyTorch offers different CUDA builds by index. Use a compatible CUDA driver for the measured build. CPU is available for functional smoke runs.

```sh
uv run --no-project .venv/bin/python benchmark.py \
  --urdf cartpole_single.urdf --output run --newton-ppo
```

A smaller smoke run keeps the same model and physics while reducing trajectory counts and optimization updates.

```sh
uv run --no-project .venv/bin/python benchmark.py \
  --urdf cartpole_single.urdf --output smoke --device cpu \
  --evaluation-cases 16 --train-worlds 32 --fit-updates 8 \
  --policy-updates 4 --policy-worlds 8 --policy-eval-cases 8 --seeds 5 --newton-ppo --skip-timing
```

To give PPO more optimization steps with the same model and seeds:

```sh
mkdir -p extended
ln -s ../run/shared-cartpole-traces.npz ../run/fit.json ../run/model.pt extended/
uv run --no-project .venv/bin/python benchmark.py \
  --urdf cartpole_single.urdf --output extended --reuse-fit --skip-timing \
  --policy-methods ppo newton_ppo --policy-updates 640
```

## Reading the receipts

- `results.json` records versions, throughput blocks, the fit curve, forward error and all policy evaluation points. Policy training seconds include reset sampling, forward rollout, backward, and optimizer work with synchronous CUDA completion. Evaluation time is recorded separately. Model fitting has separate Newton setup, data generation and training clocks. These counters exclude Python imports and initial compilation.
- `shared-cartpole-traces.npz` contains held-out passive and active states and paired actions. Each has 2,048 complete trajectories, independent of the training seed. All models receive teacher data through frame 10 for the paired forward metric. The four state values are cart position in meters, pole angle in radians and the corresponding two velocities.
- `cartpole-policy-visual.npz` stores genuine Newton replay states and normalized actions for the first 24 held-out resets, after final policy training. The held-out bank is fixed across policy seeds. Policy variability across seeds is conditional on that bank.
- `cartpole-rollout-visual.npz` stores the first 16 held-out teacher trajectories and open-loop model predictions for each scenario after frame 10.
- `model.pt` and `actor-*.pt` contain only PyTorch parameter dictionaries.

Each PPO arm uses a fixed Gaussian logit standard deviation of 0.18, four mini-batch epochs per update, clip range 0.2, GAE `lambda=.95`, and Adam learning rates of 0.001 and 0.002 for actor and critic. Pathwise Adam uses learning rate 0.01 and differentiates the full unroll. PPO uses a score-function estimator and a learned critic. They share the actor initialization, reset stream, exploration stream, batch size, rollout horizon, differentiable stage objective, and held-out Newton feedback test. Hyperparameters are fixed in the source. This pilot has no policy-search sweeps or simulator-transfer claims beyond the stated Cartpole task.

Forward timing can be repeated without training. The `newton_graph` and `learned_graph` measurements use recurrent states and a fixed action resident on the GPU. The `newton` adapter measurement additionally stages the action and reads joint coordinates into a Torch tensor. Each reports the median ms per frame from five synchronized 100-frame blocks after warmup. Kernel timings omit the policy network, resets, data generation and model training.

```sh
uv run --no-project .venv/bin/python benchmark.py \
  --urdf cartpole_single.urdf --output run --reuse-fit --timing-only
```

To regenerate the three-panel policy GIF, install Pillow and use the saved state/action coordinates. The renderer uses the first case and seed 5 by default.

```sh
uv run --no-project .venv/bin/python render_cartpole.py \
  run/cartpole-policy-visual.npz run/cartpole-policy-replay.gif
```

The policy evaluation advances all 120 frames in Newton and marks any crossing of the original safety envelope as failure. It keeps logging state costs and observations after a crossing. Reported optimizer clocks omit model commissioning and scheduled evaluation. The main paper lists the additional model setup and fit time for single-task comparisons.

## Structured controls

Fit the six-coefficient model on the same CartPole asset and held-out bank. These runs use one CPU thread.

```sh
uv run --no-project .venv/bin/python calibrate_cartpole.py \
  --urdf cartpole_single.urdf --traces run/shared-cartpole-traces.npz \
  --comparison-fit run/fit.json --output run/calibrated-modern.json

uv run --no-project .venv/bin/python benchmark.py \
  --urdf cartpole_single.urdf --output legacy-fit --legacy-inertias --fit-only
uv run --no-project .venv/bin/python calibrate_cartpole.py \
  --urdf cartpole_single.urdf --traces legacy-fit/shared-cartpole-traces.npz \
  --comparison-fit legacy-fit/fit.json --output run/calibrated-legacy.json --legacy-inertias
```

The learned correction ablation keeps the analytic mass and rail prior. The system-identification experiment fits those six coefficients from separate Newton training traces. [Structured methods and results](../index.html#methods)
