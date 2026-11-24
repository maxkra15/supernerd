# Public NeRD baseline reproduction

This extension loads the original 2,713,668-parameter NVlabs Cartpole model as an external dependency. Both public NeRD checkouts point to the same 10,895,397-byte Git LFS object. Its SHA-256 is `287dde2134eb8f5277c17a8f7cc7abbe64063e97cb481bdb6e8d76f9536e5366`. The replay script checks the revision and weight digest before loading the upstream serialized PyTorch module. The upstream code is Apache-2.0 at the repository root. [Original NeRD release](https://github.com/NVlabs/neural-robot-dynamics/tree/370e89d2b8ce68aa3578aa7812859c681b7502bc) · [Newton integration preview](https://github.com/NVlabs/neural-robot-dynamics/tree/476dc67fccc6a99be293bf611f897b1c06f13e30)

## Original Cartpole passive motion

Obtain only the Cartpole LFS model from the original release.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/neural-robot-dynamics.git nerd-original
git -C nerd-original checkout 370e89d2b8ce68aa3578aa7812859c681b7502bc
git -C nerd-original lfs pull --include='pretrained_models/NeRD_models/Cartpole/**'
```

Run with a CUDA-enabled Python environment activated. The recorded environment used Python 3.12.14, PyTorch 2.15.0+cu132 and Warp 1.8.0 on an RTX PRO 6000 Blackwell. In addition to PyTorch, the runner needs Warp 1.8.0, PyYAML, h5py, SciPy, tqdm, pyglet and OpenCV. Activate a separate environment from the modern SuperNeRD pilot and launch the original evaluator headless.

```bash
bash scripts/run_original_reference.sh nerd-original original-reference
```

This uses the authors’ published executable with 2,048 cases, seed 500 and zero action from a randomized initial state, at horizons 100, 500 and 1,000. It reports mean absolute position and angle error over trajectories and time. See Figure 3 in the original paper for the published figures. [NeRD §5.1 and Figure 3](https://arxiv.org/html/2508.15755v1#S5.SS1)

| Horizon | Published cart MAE, m | Reproduced cart MAE, m | Published pole MAE, rad | Reproduced pole MAE, rad |
| --- | ---: | ---: | ---: | ---: |
| 100 | 0.0002 | 0.000156 | 0.0004 | 0.000394 |
| 500 | 0.004 | 0.004366 | 0.013 | 0.013441 |
| 1,000 | 0.033 | 0.032776 | 0.075 | 0.075032 |

All six reproduced measurements round to the published precision. The launcher saves the upstream logs and a `replication.json` receipt containing these measurements.

## Released weights on the pilot traces

Run the replay in a separate CUDA Python environment with Warp 1.10.0, PyYAML, tqdm, PyTorch and Newton commit `668dfbb15262c0746282a8227f73337ba315f5d1`. The historic development wheel listed by that Newton checkout returned HTTP 404, so the tested setup used stable Warp 1.10.0. The selected author adapter uses Featherstone for construction and its published neural dynamics during the evaluated rollout.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch nerd_newton_dev \
  https://github.com/NVlabs/neural-robot-dynamics.git nerd-newton
git -C nerd-newton checkout 476dc67fccc6a99be293bf611f897b1c06f13e30
git -C nerd-newton lfs pull --include='pretrained_models/NeRD_models/Cartpole/**'
uv pip install 'warp-lang==1.10.0' pyyaml tqdm \
  'newton @ git+https://github.com/newton-physics/newton.git@668dfbb15262c0746282a8227f73337ba315f5d1'
for cohort in passive active; do
  uv run --no-project python scripts/official_nerd_replay.py --nerd-dir nerd-newton \
    --input run/shared-cartpole-traces.npz --cohort "$cohort" \
    --result "official-$cohort.json"
done
uv run --no-project python scripts/official_nerd_replay.py --nerd-dir nerd-newton \
  --timing-only --result official-throughput.json
```

The replay takes ten teacher steps to initialize history and compares the next 200 autoregressive frames against the supplied states for each cohort of 2,048. The replay applies the same normalized commands with the upstream 1,500 N gain and checks actual force equality. It reports per-channel RMSE and joint NRMSE, including periodic pole-angle error. Each channel is normalized by its empirical standard deviation on the supplied test traces. This differs from FlashNeRD’s training-set scaling. The eager step timing reports five blocks of 100 steps after ten warm steps at batches 1, 128 and 2,048, with inputs already on the GPU. It covers the complete public adapter and includes no policy updates.

## Imported inertia changes the task

Identical asset bytes can produce different dynamics. The original Warp import defaults to computing body mass and inertia from collision geometry and density. Under this mode, the cart mass is 50 kg and the pole mass 2.5 kg. Modern Newton honors the values written in the same URDF by default, yielding cart and pole masses of 1 kg and 10 kg respectively. Both mass sets were queried from the actual runtime builders. The modern pilot also provides a `--legacy-inertias` control which restores the original mass and center of mass under Newton. The same-mass control tests transfer across simulators with the principal known import discrepancy removed. The original evaluator is the reference for the paper’s exact passive test. [Original Cartpole loader](https://github.com/NVlabs/neural-robot-dynamics/blob/370e89d2b8ce68aa3578aa7812859c681b7502bc/envs/warp_sim_envs/env_cartpole.py) · [Modern Newton URDF importer](https://github.com/newton-physics/newton/blob/0ced8f8420d4ee1eab9ecfaee59e1354ef670348/newton/_src/utils/import_urdf.py)

Pass `--legacy-inertias --fit-only` to the pilot launcher and use its `shared-cartpole-traces.npz` as the replay input. In the original Warp 1.8.0 environment, run the analytical parity control over the same traces.

```bash
uv run --no-project python scripts/check_teacher_parity.py --nerd-dir nerd-original \
  --input legacy-fit/shared-cartpole-traces.npz --result teacher-parity.json
```

At 200 steps the original Warp analytical teacher against the same-mass modern Newton reference had endpoint NRMSE 0.0540 passive and 0.1361 active. The published NeRD model against that same reference had 0.0544 and 0.1355. The closely matched errors show that teacher disagreement dominates this transfer score. Each experiment shares the exact initial state and prescribed actions for every case.

## FlashNeRD published figures

The linked public project repository contains figures, video and page source. The authors' public artifacts found during this audit provide no runnable optimizer, learned weights or evaluation traces for their revised architecture. Their Table I is a GPU-resident, graph-replay, model-only experiment on GB200 with the NeRD and FlashNeRD models trained on a shared dataset. Their ANYmal-C time-to-target numbers include the policy optimization loop and exclude data generation, fitting the dynamics and process initialization. `published_reference.json` separates these paper values from locally measured experiments. [FlashNeRD results](https://arxiv.org/html/2610.09130v1#S4) · [Project site source](https://github.com/flashnerd/flashnerd.github.io)
