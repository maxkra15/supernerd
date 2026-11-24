# Reproduce the force-driven cable and cloth experiments

Run the commands from the `deformables` directory with NumPy and PyTorch. Fitting defaults to CUDA. The exporter supports CPU and CUDA. The native-vertex scorer runs on CPU. The bundled checkpoints contain learned weights and fit-set normalization.

## Score the released model

```sh
mkdir -p replay
python export_forward.py --device cuda --model weights/modal-paired.pt --data data/fresh.npz --output replay/fresh.npz
python native_rollouts.py score data/fresh.npz data/native-fresh.npz replay/fresh.npz replay/native-fresh.json
python verify_forward.py --weights weights/modal-long.pt --data data/test.npz --output replay/stream-causality.json
```

On the recorded GPU, replay with `--device cuda` reproduced all saved positions exactly. With `--device cpu`, mean native errors differed by at most 0.014 mm in these four family and restart comparisons.

Substitute `data/test.npz` and `data/native-test.npz` to score the original evaluation set. The exporter writes trajectory IDs and normalized shape coordinates in matching order. From frame zero there are 300 predictions at 20 ms intervals. From frame 100 there are 200. Each rollout starts from one observed shape and velocity. Each subsequent step consumes the current prescribed tip or edge acceleration. The neural state carries forward without teacher resets.

## Refit

```sh
python modal_forward.py --data data --output replay/short --updates 1000 --budgets 8 32 192
python modal_forward.py --data data --output replay/long --updates 8000 --budgets 192
python paired_forward.py --data data --output replay/paired --steps 8000
```

The first command fits three independent models on 8, 32, and 192 objects per material. The second uses 192 per material for a longer optimization. Both use one diverse load per training object. The third adds a second Newton trajectory for every object and a loss on shape differences between load programs. Its 768 training trajectories represent 384 unique objects. The paired experiment was designed after inspecting the original results. Fresh geometry seeds provide a separate test of the revised recipe. Each test contains eight objects per material and five load conditions per object; these conditions share the same initial state.

A shared encoder maps the initial shape, velocity, geometry, and material descriptors to 32 damped complex oscillators, forcing gains and a fixed readout context. Each oscillator advances by a rotation, decay and forcing update. A shared nonlinear decoder returns XYZ positions at fifteen sample locations. Clamped-edge constraints are imposed explicitly. Cloth uses a 3-by-5 sample grid. Cable uses five axial points, repeated across the lateral dimension. Native evaluation interpolates these samples to each original teacher vertex and measures the resulting error, including spatial detail lost during sampling.

## Newton data

The distributed data can be used directly. The archived training programs and random seed ranges are below. Each command writes a new raw trajectory directory. Newton and Warp revisions and a timed repeat of all 384 primary training captures are recorded in [the capture receipt](results/teacher-capture.json). [The pack manifest](data/provenance.json) gives compact data hashes and feature names.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m teacher.elastic_load_diversity --seed 24500 --cases 192 --program train-diverse --output raw/load-diversity-v5-diverse
python -m teacher.elastic_load_diversity --seed 24500 --cases 192 --program train-pulses --output raw/pulses
python -m teacher.elastic_load_diversity --seed 30200 --cases 16 --program train-diverse --split development --output raw/load-diversity-v5-development
python -m teacher.elastic_load_diversity --seed 40500 --cases 8 --program held-composition --output raw/load-diversity-v5-held
python -m teacher.elastic_load_diversity --seed 41540 --cases 8 --program held-composition --output raw/fresh
python pack_deformables.py raw replay/data
python paired_forward.py --data replay/data --paired-source raw/pulses --fresh-source raw/fresh --output replay/regenerated-fit --steps 8000
```

Training and GPU timings measure optimization on packed trajectories. The teacher capture receipt times simulation and file output separately. Both modalities use gravity and applied loading. These experiments evaluate a fixed support and free-space motion; they leave frictional contact, self-collision and independently moving supports untested.

To repeat the single-core online latency benchmark, run `python matched_forward_timing.py --output replay/latency.json` on Linux with a pinned Newton installation. This includes native-position reconstruction for each frame.
