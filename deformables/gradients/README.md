# Training a dynamics surrogate with force derivatives

The experiments compare ordinary rollout backpropagation, supervision from two observed responses, and a loss on the model's exact directional derivative with respect to a waveform amplitude. The native teacher is Newton VBD for cable and cloth and Newton Featherstone for CartPole. [Read the scientific report](https://maxkra15.github.io/supernerd/#gradients).

## Training and evaluation data

The original shared model and training environment are documented in [the parent experiment](../MODAL_REPRODUCTION.md). New data under [data](data/) are synthetic CPU Newton observations. The data contain 64 adaptation objects (32 of each material) and 16 independent sensitivity probes. Each training object has a nominal trajectory and a pair with amplitudes ±0.5 m/s². Evaluation adds ±0.25 m/s² for a step-halving check and +3 m/s² to assess finite responses. Each six-second rollout starts from the same material-specific fixed release. Eight previously unused seeds per material, 44810–44817, form the larger triangle/chirp confirmation bank. Each seed has five force schedules. All stored observations include their object identity, and file fingerprints are listed in [sha256.json](data/sha256.json).

The timed training-only capture wrote 192 trajectories into 64 object archives, all byte-identical to the original training archives in this release. [Timing and verification](results/fit-capture.json). These measurements exclude creating the pretrained parent. Model fitting and Newton capture are timed separately.

## Deformable experiments

Use the environment pinned in the main report. Run the commands below **from the deformables directory**. The original 28.9 s shared checkpoint is in weights/modal-paired.pt. The GPU experiments ran one job at a time on a single local RTX PRO 6000 Blackwell. CPU runs use a single thread for model fitting.

~~~sh
python -m gradients.force_gradient_trial --data data --output replay/paired
python -m gradients.secant_check --data data --weights weights/modal-paired.pt --output replay/curvature.json

python -m gradients.local_fit --data gradients/data --output replay/local
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gradients.readout_fit --data gradients/data --parent weights/modal-paired.pt --output replay/readout

python -m gradients.retention fit --data data --local gradients/data --parent weights/modal-paired.pt --output replay/retention
python -m gradients.retention score --data data --local gradients/data --parent weights/modal-paired.pt --output replay/retention --confirmation-data gradients/data/confirm.npz --confirmation-native gradients/data/confirm-native.npz
~~~

To score just the released continuation on the 80 new trajectories, run these commands. CPU output is the scored endpoint used in the follow-up study.

~~~sh
mkdir -p replay
python export_forward.py --model gradients/results/retention/mixed-271.pt --data gradients/data/confirm.npz --output replay/confirm.npz
python native_rollouts.py score gradients/data/confirm.npz gradients/data/confirm-native.npz replay/confirm.npz replay/confirm.json
python motion_figure.py --packed gradients/data/confirm.npz --native gradients/data/confirm-native.npz --predictions replay/confirm.npz --seed 44810 --output replay/confirmation.gif
python -m gradients.plot_response
~~~

To regenerate the local perturbations, run the CPU capture. To regenerate the independent triangle-and-chirp bank, use the existing teacher. The original recordings have a 20 ms output interval, four physics substeps per interval and eight VBD iterations per substep.

~~~sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m gradients.teacher_probes --output replay/probes --workers 8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m teacher.elastic_load_diversity --program held-composition --seed 44810 --cases 8 --workers 8 --output replay/confirm-raw
python -c 'from pathlib import Path; from paired_forward import repack; repack(Path("replay/confirm-raw"), Path("replay/confirm-packed.npz"))'
python native_rollouts.py pack replay/confirm-raw replay/confirm-native.npz
~~~

## Interpret the comparisons

- **Original waveform pairs.** Both networks train from scratch on the same initialization and minibatches. On windows starting at release, the perturbed trajectories share the same pose and velocity. The JVP arm compares the exact midpoint derivative of the neural rollout to the observed Newton difference between finite waveform changes. Remaining restart windows keep the usual paired loss. This is one training seed. The matched baseline and the earlier published fit are separate runs. [Matched update and wall-time results](results/paired/results.json).
- **Small perturbations.** Three fine-tuning seeds share the same checkpoint and 192 measured trajectories. Each fit runs 1,200 updates. One loss uses all three position records. The other two also compare a predicted slope against the same central Newton estimate. The secant uses two model outputs. Autodiff computes the exact directional derivative of the differentiable model at the nominal program, then backpropagates its training loss through this derivative. The test compares against the finer central estimate over all native vertices after force onset. Every test object has equal weight. [Metrics and checkpoints](results/local/).
- **Frozen readout.** The two fits use identical frozen latent features and one shared linear output across materials. A float64 ridge system fits changes to free-point weights. Its fixed regularizer is 0.01 times the mean diagonal of the position feature covariance. The derivative term in the second objective is scaled to a 3 m/s² force change. The readout timings include extraction, fit statistics and solving, after one warm-up object. They exclude data loading, Newton data collection, evaluation and parent training. [Results](results/readout/).
- **Continuation with mixed trajectories.** Both comparison arms see the same original paired windows at each of 1,200 updates. The control continues training on the original objective. The mixed model adds a second batch of 12 objects with small perturbations at each update. Its loss weights the original objective by 0.75 and the new shape-and-secant objective by 0.25. The two minibatch-order seeds start from the same pretrained weights. The updates and the order of original examples match across arms. Mixed updates require extra compute and measurements. We chose this follow-up in response to a loss of transfer from local-only training. All confirmation trajectories remained separate from fitting. [Per-case results, weights and training clocks](results/retention/).

## Robot experiment

From the repository root, obtain the pinned URDF with the command in [the CartPole reproduction guide](../../paper/cartpole-reproduction.md). Run with the public robot reference weights already in the repository.

~~~sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m deformables.gradients.robot_cartpole --urdf cartpole_single.urdf --output replay/robot
~~~

All three residual MLP arms see 6,144 center transitions. Two arms also see the same 12,288 one-step force perturbations (±6 N). One directly learns the perturbed states. The other matches the four force derivatives obtained by differentiating its learned one-step transition. Three initializations and matched four-step rollout losses are compared. The 192 held-out forced and 192 passive trajectories each last 100 steps. Derivatives are scored on 1,152 new state-action pairs, with a 1.5 N evaluation half-width. Doubling the half-width changed the reference by less than 0.15% on each component. [Robot metrics](results/robot/results.json) · [Full-budget MLP reference](results/robot/reference-mlp.json).
