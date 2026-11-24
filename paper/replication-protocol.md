# Replication protocol

## Questions

1. Can a small differentiable transition fit 200-step open-loop trajectories on the same CartPole asset?
2. With the learned transition, observation, reward and actor fixed, does backpropagation through time improve Newton performance at the same imagined-transition or wall-clock budget?
3. Do the first two findings extend to contacts and locomotion under the published FlashNeRD protocols?

## CartPole policy comparison

Use the public single-pendulum CartPole asset from the NeRD Newton branch. Log simulator transitions using policies that apply randomized open-loop forces from both broad states and near-upright states. Fit from disjoint trajectory groups. Use a small residual MLP in generalized coordinates. A held-out dynamics set measures recursive state error at 1, 15, 100 and 200 decisions. A separate reset distribution tests transfer of control.

Use the same frozen learned dynamics and initial actor initialization for clipped PPO and policy gradients backpropagated through imagined rollouts. Log simulator transition count and synchronized training time. Evaluate each recorded checkpoint on the same Newton resets and use deterministic actions. Report costs and success by seed, and keep data generation and dynamics fitting times separate from policy optimization time. A speedup requires reaching comparable Newton quality.

The public original NeRD checkpoint has its own supported backend and timing configuration. Report compatibility and standalone rollout results before using it in a head-to-head comparison. Asset, coordinate convention, reset history and control cadence must all match.

## FlashNeRD replication targets

| Target | Published protocol | Requirements for an equivalent comparison |
| --- | --- | --- |
| CartPole transition | Newton Featherstone labels, ten history frames, roughly 9,000 trajectories, 2,048 evaluation trajectories, 1/15/100/200-step prefix NRMSE | Original trajectories or independently validated identical splits, normalization and transition cadence. Compare the same input preparation, dtypes and CUDA graph modes at the same batch sizes. |
| ANYmal-C velocity tracking | Newton MJWarp evaluation, 16,384 training environments, PPO 300 × 24 decisions, four foot sphere probes | Match robot and actuator, reset/control distribution, reward and observation, contact handling, and hardware. Track both training return and Newton return at equal time, interactions and total compute. Add backpropagation through time on the same learned transition. |
| G1 rollout and contact control | G1 transition accuracy with Newton MJWarp labels and eight fixed probes in the model-only benchmark | Start with the same supervised transition task. Measure held-out transfer over contact onsets, foot slip and contact loss before attaching a tracking policy. |
| Allegro cube reorientation | Robot free dynamics, Newton MJWarp reaction forces, 512 held-out episodes, 10 s cap | Use the same two-way constraint solve, episode count and target definition. Verify that a gradient update can observe changes in the contact solve. |

Published baselines are references until a matched run has been executed. See the [FlashNeRD experimental sections](https://arxiv.org/html/2610.09130v1#S4).
