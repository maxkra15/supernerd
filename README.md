# SuperNeRD

SuperNeRD learns how a deformable object moves under a given force. One forward model shares its weights across cables and cloth. The primary experiment measures the Newton data required, time to fit, and shape and force-response errors over complete unseen trajectories.

[Read the paper](https://maxkra15.github.io/supernerd/) · [PDF](paper/deformable-physics.pdf) · [Watch Newton and SuperNeRD side by side](figures/newton-supernerd.gif)

On 16 fresh objects and 80 six-second trajectories, the shared model averages **1.57% of cable length and 4.28% of cloth length** in native-vertex position error. Its 8,000-update fit took **28.9 seconds on one GPU**. Generating the 768 training trajectories took another **168.9 seconds on eight CPU workers** across two timed exports with warm kernels. On the smaller motion caused by changing just the force, the mean relative error is **55.6% for cable and 89.8% for cloth**. These tests use fixed supports and free-space deformation. At batch one on a single CPU core, updating native-vertex positions was 112× faster than Newton for cable and 139× for cloth. [Matched timings](deformables/results/matched-forward-timing.json).

![A six-second Newton and SuperNeRD rollout on a fresh cable and cloth under a descending chirp](figures/newton-supernerd.gif)

## Task performance in Newton

SuperNeRD and NeurALL now choose actions on the same cable task. The released SuperNeRD cable passed both timed tip gates on **16/32** goals, NeurALL on **14/32**, and a three-probe continuous Newton controller on **32/32**. The gates move substantially under teacher refinement. On tabletop cloth, a new learned model ranked 12 grasp motions. Checking its top three in Newton completed **14/16** folds, against **15/16** for a full search, at **3.91×** the planning speed on one CPU core. The learned choice alone completed **9/16**. [Task results and videos](https://maxkra15.github.io/supernerd/#tasks) · [Cable benchmark](tasks/cable/README.md) · [Folding benchmark](tasks/cloth/README.md).

## Force-response learning

Starting from the released model, an additional 7.1 seconds of training on original trajectories and new force perturbations improved release rollouts on an independent 80-trajectory confirmation set. Cable shape error fell from 1.41% to 1.28%, and cloth force-response error fell from 95.04% to 88.59%. A loss that explicitly differentiates the surrogate gave similar accuracy to ordinary backpropagation on measured displacement differences and took longer. Moving cloth restarts remain a weakness. [Study](https://maxkra15.github.io/supernerd/#gradients) · [New rollout GIF](figures/gradient-confirmation.gif) · [Data, checkpoints and commands](deformables/gradients/README.md).

## Reproduce

- [Shared model, replay from stored weights, and teacher generation](deformables/MODAL_REPRODUCTION.md)
- [Streaming attention model and cache timing](deformables/STREAM.md)
- [Calibration from a few force programs on a single asset](deformables/asset/)
- [Training and held-out trajectories](deformables/data/), [checkpoints](deformables/weights/), [measurements](deformables/results/)
- [Original NeurALL experiments](https://maxkra15.github.io/neurall/) and [archived CartPole policy study](https://maxkra15.github.io/supernerd/cartpole.html)

To render the paper with a uv environment containing Jinja2, NumPy and Matplotlib, and a local Google Chrome installation for PDF export, run the following from the repository root.

~~~sh
uv run --no-project .venv/bin/python deformables/motion_figure.py
uv run --no-project .venv/bin/python scripts/build_deformables.py --pdf
~~~

SuperNeRD's original code is released under the [Apache License 2.0](LICENSE).
