# Streaming forward dynamics for cable and cloth

A single 40,272-parameter model advances the 15 canonical point positions and velocities using the prescribed load, material properties and geometry. The model has two width-48 blocks. Each block applies multi-query temporal attention (two query heads and one shared key/value head) and a SiLU-gated feedforward branch in parallel from the same projected features. Each block keeps the preceding two frames, giving a five-frame receptive field across the two blocks. A direct affine readout and learned features predict corrections to damped velocity and trapezoidal integration. Hard clamps pin the three root samples, and the three lateral samples coincide for a cable. These are contact-free trajectory benchmarks.

Each trajectory is initialized from observed state, with an empty cache at frame 0 or frame 100. Subsequent steps use predictions and prescribed loads. Fitting uses randomized teacher windows and then short recursive losses. Selection uses only development-set geometry error. The test has 40 trajectories per material, with five commands per held-out geometry.

## Reproduce

The scripts require the canonical train.npz, development.npz and test.npz packs, PyTorch and NumPy. Run the following commands from the directory containing the scripts, with a Python environment containing PyTorch, NumPy and a supported CUDA device. Set the Python executable to your environment's interpreter if needed. Each initial training budget starts with random weights.

~~~sh
uv run --no-project python stream_deformables.py \
    --data data --output stream --budgets 8 32 192 --updates 1500
uv run --no-project python stream_deformables.py \
    --data data --output stream-extended --budgets 192 \
    --resume-model stream/stream-192.pt --updates 6000 \
    --recursive-steps 12 --lr 0.00045
uv run --no-project python stream_timing.py \
    --data data --weights stream-extended/stream-192.pt \
    --output stream-extended/stream-timing.json
uv run --no-project python native_rollouts.py score \
    data/test.npz data/native-test.npz stream-extended/stream-predictions.npz \
    stream-extended/native-stream-long.json
~~~

| Training stage at 192 trajectories per family | Fit seconds | Cable, 300 steps | Cloth, 300 steps | Cable, moving restart | Cloth, moving restart |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial 1,500 updates | 3.39 | 9.12% | 12.00% | 6.30% | 9.24% |
| Additional 6,000 updates | 47.82 cumulative | 4.06% | 4.71% | 2.26% | 4.37% |

The error is the Euclidean position RMS across the 15 canonical points divided by rest length, averaged over predicted frames and trajectories. The moving restart covers 200 decisions at 0.02 s. Both stages together required 51.39 s including preprocessing, optimizer setup and scheduled validation. These measurements use cached Newton labels and a shared RTX PRO 6000 Blackwell. The initial full-budget fit ran after the smaller budgets in the same process.

Cache reuse reduces GPU forward latency relative to recomputing the full five-frame receptive field. The full and cached paths use the same parameters and FP32 CUDA graph replay, including maintenance of the temporal window. Median speed ratios were 1.32×, 1.31× and 1.15× at batches 1, 80 and 512. The implementation uses ordinary PyTorch operators and a fixed-length tensor cache. FlashNeRD additionally provides specialized kernels, a learned contact-set encoder and hybrid two-way analytical coupling. [FlashNeRD method](https://arxiv.org/html/2610.09130v1#S3)

The timing comparison measures rollout latency. The fast-fit comparison uses complete held-out autonomous geometry error. Estimating how much attention itself helps would require a trained architecture ablation on an identical training protocol.
