"""Assemble the paper and normalized figure data from the run receipts."""

from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HORIZONS = ("1", "15", "100", "200")
COHORTS = ("passive", "active")
STATE_KEYS = ("cart_position_m", "pole_angle_rad", "cart_velocity_m_s", "pole_angular_velocity_rad_s")
ARM_NAMES = {"pathwise": "Backpropagation · learned", "ppo": "PPO · learned", "newton_ppo": "PPO · Newton"}


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def paragraph(text: str) -> str:
    return f"<p>{text}</p>"


def figure(path: str, label: str, description: str, caption: str) -> str:
    return (
        f'<figure><img src="figures/{path}" alt="{escape(description)}">'
        f"<figcaption><strong>{label}.</strong> {caption}</figcaption></figure>"
    )


def table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f'<th scope="col">{escape(col)}</th>' for col in headers)
    body = []
    for row in rows:
        values = "".join(f'<td class="num">{escape(value)}</td>' for value in row[1:])
        body.append(f'<tr><th scope="row">{escape(row[0])}</th>{values}</tr>')
    contents = "".join(body)
    return (
        '<div class="table-wrap"><table>'
        f'<caption>{caption}</caption><thead><tr>{head}</tr></thead>'
        f'<tbody>{contents}</tbody></table></div>'
    )


def normalized(data: dict, calibration: list[float]) -> float:
    return stats.fmean((data[key] / scale) ** 2 for key, scale in zip(STATE_KEYS, calibration, strict=True)) ** 0.5


def rollout_data(run: dict, structured: dict) -> dict:
    calibration = run["model"]["train_state_std"]
    result = {}
    calibrated = read("calibrated-modern-common.json")["rollout_common_normalizer"]
    for cohort in COHORTS:
        original = read(f"nerd-{cohort}.json")["prefix_and_terminal_error"]
        result[cohort] = {
            "SuperNeRD MLP": {h: run["model"]["rollout"][cohort][h]["prefix_nrmse"] for h in HORIZONS},
            "Released NeRD checkpoint": {h: normalized(original[h]["prefix_rmse"], calibration) for h in HORIZONS},
            "Structured prior": {h: structured["zero_residual"][cohort][h]["prefix_nrmse"] for h in HORIZONS},
            "Prior + force residual": {h: structured["learned_residual"][cohort][h]["prefix_nrmse"] for h in HORIZONS},
            "Calibrated mechanical prior": {h: calibrated[cohort][h]["prefix_nrmse"] for h in HORIZONS},
        }
    return result


def policy_table(run: dict) -> str:
    rows = []
    for key, method in run["policy"]["arms"].items():
        finals = [seed["curve"][-1] for seed in method["seeds"]]
        successes = [p["newton"]["success_rate"] * 100 for p in finals]
        costs = [p["newton"]["mean_native_cost"] for p in finals]
        times = [p["training_s"] for p in finals]
        rows.append([
            ARM_NAMES[key], f"{stats.fmean(successes):.1f}%",
            f"{min(successes):.1f}–{max(successes):.1f}%",
            f"{stats.fmean(costs):.3f}", f"{min(times):.2f}–{max(times):.2f}",
        ])
    return table(
        ["Optimizer", "Mean success", "Seed range", "Mean cost", "Training time (s)"],
        rows,
        "Table 1. Final policies at 294,912 training transitions per seed. Each seed is evaluated on the same 256 "
        "Newton resets. Wall time covers the optimizer and data collection for policy updates.",
    )


def timing_table() -> str:
    rows = []
    for batch, modes in read("forward-timing.json").items():
        newton, eager, graph = (modes[name]["ms_p50"] for name in ("newton_graph", "learned_eager", "learned_graph"))
        rows.append([batch, f"{newton:.3f}", f"{eager:.3f}", f"{graph:.3f}", f"{newton/graph:.2f}×"])
    return table(
        ["Batch", "Newton graph (ms)", "Learned eager (ms)", "Learned graph (ms)", "Newton / learned graphs"],
        rows,
        "Table 2. Median latency per 60 Hz control step, from five synchronized groups of 100 calls "
        "on one RTX PRO 6000 Blackwell. Inputs stay on device. Newton uses five physics substeps.",
    )


def extended_policy_text() -> str:
    file = RESULTS / "cartpole-extended.json"
    if not file.exists():
        return ""
    data = json.loads(file.read_text())
    rows = []
    for key, method in data["policy"]["arms"].items():
        finals = [seed["curve"][-1] for seed in method["seeds"]]
        hits = [
            next((point for point in seed["curve"] if point["newton"]["success_rate"] >= 0.90), None)
            for seed in method["seeds"]
        ]
        successes = [p["newton"]["success_rate"] * 100 for p in finals]
        times = [p["training_s"] for p in hits if p is not None]
        transitions = [p["simulator_steps"] for p in hits if p is not None]
        rows.append([
            ARM_NAMES[key], f"{finals[0]['simulator_steps']:,}",
            f"{min(successes):.1f}–{max(successes):.1f}%",
            f"{len(times)}/{len(hits)}",
            f"{min(transitions):,}–{max(transitions):,}" if transitions else "—",
            f"{min(times):.2f}–{max(times):.2f}" if times else "—",
        ])
    return (
        paragraph("An extended run used the same PPO settings and seeds with a higher fixed update budget. "
                  "A threshold crossing is the first observed checkpoint with at least 90% Newton success. "
                  "The exact crossing between evaluations is interval-censored.")
        + table(
            ["Optimizer", "Final transitions", "Final success range", "Seeds crossing 90%", "Transitions to 90%", "Seconds to 90%"],
            rows, "Table 3. Extended policy training. Times are policy updates only.",
        )
    )


def methods() -> str:
    return """
<h3>3.1 Learned decision step</h3>
<p>The model state is cart translation, pole angle and their velocities. The input is seven features from
scaled state, sine and cosine of the pole angle, normalized force and signed proximity beyond the rail-limit
margin. A 4,932-parameter MLP with two width-64 SiLU layers outputs four state corrections over a kinematic
position update. The architecture receives a single observed state. The force remains constant across the
five Newton physics substeps in each label.</p>
<div class="equation" aria-label="Next state equals kinematic prediction plus learned residual">
  ŝ<sub>t+1</sub> = [q<sub>t</sub> + h v<sub>t</sub>, v<sub>t</sub>] + D f<sub>θ</sub>(φ(ŝ<sub>t</sub>, u<sub>t</sub>))
</div>
<p>The diagonal scale <em>D</em> puts the four residual components on comparable scales for fitting.
The first 80% of updates use one-step error. The final 20% unroll four predictions with the recorded forces.
The loss is the mean squared error of state normalized by component and horizon. Newton supplies the labels
and accepts replay actions through the same force interface.</p>
<h3>3.2 Policy update</h3>
<p>All arms use the same actor and initial observation and exploration distributions. The actor outputs a
Gaussian logit mean with fixed standard deviation 0.18 in training. A tanh maps the sampled logit to a cart
force bounded at ±225 N. Training costs include pole-angle error, cart displacement, joint speeds, force and
soft state-limit penalties. Evaluation takes the mean logit and uses the stage cost for the physical state.</p>
<div class="equation" aria-label="Policy gradient through recursive transitions">
  J(ψ) = E<sub>s₀,ε</sub>[H<sup>−1</sup> ∑<sub>t=0</sub><sup>H−1</sup> ℓ(ŝ<sub>t+1</sub>, u<sub>t</sub>)]
  &ensp; with &ensp; ŝ<sub>t+1</sub> = F<sub>θ</sub>(ŝ<sub>t</sub>, u<sub>ψ,t</sub>).
</div>
<p>Backpropagation takes the derivative of this objective through 48 imagined steps with frozen dynamics.
The PPO arms fit a critic and use clipped updates with ratio clipping of 0.2 and GAE with λ=0.95.
One PPO arm samples the frozen model. The other uses Newton. Actor learning rates are 0.01 for
backpropagation and 0.001 for PPO. The comparison measures these optimization recipes at equal
environment interactions, with wall time reported separately. <a href="#ref-ppo">[4]</a></p>
<h3>3.3 Mechanically structured comparison</h3>
<p>A specialized analytical prior integrates a two-coordinate mass matrix, gravity, centrifugal bias
and stiff rail-limit forces for five semi-implicit substeps. It refreshes the inertia once per decision,
as configured in Newton. A second model adds a 322-parameter MLP residual to generalized forces, with its final
layer initialized to zero. The residual test uses the same training distribution and held-out trajectories.
A separate system-identification control fits six scalar coefficients from Newton transitions. It begins
with deliberately perturbed mass, inertia, coupling, gravity, rail stiffness, and damping. Positive
inertia parameters and a bounded coupling maintain a positive-definite mass matrix. We fit the same
formula separately under the two inertia conventions.</p>
"""


def setup(run: dict) -> str:
    return f"""
<p>The public NeRD CartPole URDF is evaluated with Newton Featherstone at a 60 Hz decision rate.
Training data are {run["model"]["train_transitions"]:,} transitions from 4,096 Newton worlds started from
a mixture of broad and near-upright states with randomized open-loop forces. Each held-out open-loop
cohort has 2,048 reset trajectories, passive or actively forced. Autonomous prediction is scored over the
200 steps after the first ten observed frames. The released NeRD model uses all ten observations. The primary control benchmark uses the inertias declared by the URDF, with 1 kg and 10 kg cart and pole masses.
Checkpoint, asset and run metadata are versioned with the report.</p>
<p>For each policy seed (5, 19, 47), a policy update uses 64 reset worlds for 48 decisions. Checkpoints are
tested with deterministic feedback over 120 Newton decisions and a separate 256-reset seed.
Success requires the cart and speeds to stay inside the declared state limits and the
upright angle and angular velocity to stay within 0.15 rad and 0.5 rad/s for all final 60 decisions.
The 90% success threshold is measured on these Newton replays.</p>
<p>Open-loop error uses the root mean square over trajectories and the indicated prefix, with pole-angle
differences wrapped. Each state channel is normalized by its training-trajectory standard deviation
before the root mean square across channels. Endpoint and one-step teacher-forced errors are retained
in the raw receipts. This normalization and the test data are shared across the local figure.</p>
"""


def results_text(run: dict, rollouts: dict) -> str:
    mlp_h200 = rollouts["passive"]["SuperNeRD MLP"]["200"]
    prior_h200 = rollouts["passive"]["Structured prior"]["200"]
    learned_h200 = rollouts["passive"]["Prior + force residual"]["200"]
    mlp_grad = read("action-gradients.json")["results"]
    structured_grad = read("action-gradients-structured.json")["results"]
    grad_h32 = mlp_grad["32"]["near_upright"]
    struct_h32 = structured_grad["32"]["near_upright"]
    fit = run["model"]
    commissioning = sum(fit[key] for key in ("teacher_setup_s", "teacher_rollout_s", "fit_s"))
    return (
        "<h3>5.1 Policy quality and time</h3>"
        + paragraph("Backpropagation first reaches at least 90% success in all three seeds at the update-24 "
                    "checkpoint, after 73,728 imagined transitions and 1.13–1.73 s of policy training. "
                    "It reaches 100% in all seeds at 294,912 transitions. The learned-model PPO arm ends "
                    "at 7.4–16.0% and Newton PPO at 10.9–20.7% at that budget. At equal steps, "
                    "backpropagation takes longer per update. The quality advantage drives its sample efficiency.")
        + policy_table(run)
        + figure("policy-learning.svg", "Figure 1", "Success and cost versus training transitions and time",
                 "Thin lines are individual policy seeds. Thick lines are seed means on the common support. "
                 "Each point is a separate analytical Newton policy replay. Lower cost is better.")
        + extended_policy_text()
        + figure("extended-policy-learning.svg", "Extended training", "PPO improvements at longer budgets",
                 "Same PPO settings and seeds as the fixed-budget comparison. Both PPO arms eventually "
                 "reach 100% Newton success across all three seeds.")
        + paragraph("Median policy-training time to the 90% threshold is 5.90× lower than learned-model PPO "
                    "and 5.53× lower than Newton PPO. These ratios use medians across the three seeds. "
                    "With dynamics setup, data generation, fitting and scheduled Newton evaluation charged, "
                    "medians are 4.69 s for backpropagation, 11.19 s for learned-model PPO and 7.46 s "
                    "for Newton PPO. The longer PPO runs also reach 100% and achieve lower stage cost "
                    "than the initial 96-update backpropagation runs. A separate 320-update backpropagation run "
                    "keeps 100% success but still ends at higher median cost (0.0414) than learned PPO at 640 "
                    "(0.0281). The longer backpropagation runs are scored by quality here. "
                    '<a href="results/threshold-summary.json">Threshold accounting.</a>')
        + paragraph(f"The learned arm uses {commissioning:.2f} s to create the training worlds, log training "
                    "rollouts and fit the dynamics after caches are warm. Adding that cost to the pathwise "
                    "threshold times gives about "
                    f"{commissioning + 1.13:.2f}–{commissioning + 1.73:.2f} s, before replay evaluations. "
                    "This separates an upfront world-model fit from amortized policy learning.")
        + '<figure><img src="figures/cartpole-policy.gif" alt="Sample CartPole policy actions replayed in Newton">'
          '<figcaption><strong>Rollout.</strong> The same fixed held-out start is replayed with mean actions '
          'from the three final policies. All trajectories were generated by Newton.</figcaption></figure>'
        + "<h3>5.2 Prediction and throughput</h3>"
        + paragraph(f"Passive prefix NRMSE reaches {mlp_h200:.3f} at 200 steps for the unrestricted MLP. "
                    f"The mechanical prior gives {prior_h200:.2e} on the same trajectories. Adding the "
                    f"trained force residual raises that to {learned_h200:.2e}. The structured prior already "
                    "captures this teacher closely, so fitting a free correction can amplify small errors.")
        + figure("rollout-errors.svg", "Figure 2", "Normalized rollout error for two held-out control cohorts",
                 "The original NeRD checkpoints were fitted to legacy Warp labels. Local normalized scores "
                 "for those weights measure transfer to the current Newton teacher.")
        + paragraph("The six-coefficient calibration runs in 49–75 ms on CPU after 0.50–0.58 s of "
                    "teacher setup and data generation. With the corresponding neural fit normalizers, "
                    "its passive and active 200-step prefix NRMSE are 3.75e−6 and 9.81e−4 for the modern "
                    "asset and 3.91e−5 and 4.47e−5 under legacy inertias. The known CartPole equations "
                    "and rail force law supply the structure for this comparison. "
                    '<a href="results/calibrated-modern-common.json">Modern fit</a> · '
                    '<a href="results/calibrated-legacy-common.json">Legacy fit</a>.')
        + timing_table()
        + "<h3>5.3 Action derivative fidelity</h3>"
        + paragraph("We differentiate a separate smooth tracking objective through 1, 8 and 32 controls "
                    "for 16 near-upright and 16 wide-state cases. Central differences check PyTorch derivatives "
                    "and independently estimate Newton derivatives at two force perturbation sizes. "
                    f"Near upright at 32 steps, the median cosine with Newton is {grad_h32['median_cosine']:.3f} "
                    f"for the unrestricted model and {struct_h32['median_cosine']:.3f} for the structured "
                    f"residual. Median relative gradient-vector errors are {grad_h32['median_relative_gradient_error']*100:.2f}% "
                    f"and {struct_h32['median_relative_gradient_error']*100:.3f}%, respectively. All 16 near-upright "
                    "cases improve with one fixed-size negative-gradient step for either model.")
        + figure("action-gradients.svg", "Figure 3", "Gradient agreement and actual Newton descent of the learned model",
                 "Dots are individual cases and diamonds are medians. Positive cost reduction means the "
                 "learned action step improved the objective when evaluated by Newton.")
        + paragraph('<a href="figures/action-gradients-structured.svg">Structured model derivative plot</a> · '
                    '<a href="results/action-gradients.json">MLP probe data</a> · '
                    '<a href="results/action-gradients-structured.json">Structured probe data</a>')
    )


def release_text() -> str:
    data = read("nerd-passive.json")
    src = data["nerd_revision"]
    elapsed = data["checkpoint_sha256"]
    original = read("nerd-original.json")
    reported = read("published_reference.json")["neural_robot_dynamics"]["published_cartpole_passive"]
    rows = []
    for i, horizon in enumerate(reported["horizons"]):
        row = original["mean_absolute_error_over_time_and_cases"][str(horizon)]
        rows.append([
            str(horizon), f"{row['cart_m']:.6f}", f"{reported['cart_mae_m'][i]:g}",
            f"{row['pole_rad']:.6f}", f"{reported['pole_mae_rad'][i]:g}",
        ])
    original_table = table(
        ["Horizon", "Cart replay (m)", "Cart paper (m)", "Pole replay (rad)", "Pole paper (rad)"],
        rows, "Original NeRD Figure 3 passive CartPole mean absolute error. "
        "The author script, original checkpoint, Warp 1.8.0 and the 2,048-case seed-500 protocol reproduce all six rounded values.",
    )
    legacy_fit = read("legacy/fit.json")
    legacy_teacher = read("legacy/teacher-parity.json")["cohorts"]
    legacy_rows = []
    for cohort in COHORTS:
        original_neural = read(f"legacy/official-{cohort}.json")["prefix_and_terminal_error"]["200"]
        refit = legacy_fit["rollout"][cohort]["200"]
        legacy_rows.append([
            cohort.capitalize(), f"{original_neural['terminal_nrmse']:.4f}",
            f"{refit['endpoint_nrmse']:.4f}", f"{legacy_teacher[cohort]['200']['terminal_nrmse']:.4f}",
        ])
    legacy_table = table(
        ["Test cohort", "Released NeRD", "Refit SuperNeRD MLP", "Original Warp teacher"],
        legacy_rows, "200-step endpoint NRMSE after matching original masses in the current Newton teacher. "
        "All rows use the held-out Newton states and actions. Errors in the analytical Warp column quantify "
        "cross-simulator drift independently of the networks.",
    )
    return (
        paragraph("We reproduced original NeRD Figure 3 on its legacy Warp teacher and public checkpoint. "
                  '<a href="results/nerd-original.json">Replay receipt.</a>')
        + original_table
        + paragraph(
            f"The released {data['model_parameters']:,}-parameter NeRD model also ran on all local active "
            "and passive trajectories with ten teacher frames of warmup and the same recorded forces. "
            "The legacy Warp importer infers 50 kg and 2.5 kg cart and pole masses from collision geometry. "
            "The primary current Newton teacher uses the URDF-declared 1 kg and 10 kg inertias. "
            f'<a href="https://github.com/NVlabs/neural-robot-dynamics/tree/{src}">Pinned NeRD source</a> · '
            f"<code>{elapsed}</code> checkpoint SHA-256."
        )
        + paragraph("A second Newton run uses the legacy-inferred masses and re-fits SuperNeRD from scratch. "
                    "Released NeRD is substantially more accurate than the small learned MLP in both "
                    "cohorts. The original analytical Warp solver incurs similar error to the released "
                    "checkpoint, meaning different numerical solvers still contribute an error of that scale "
                    "after inertia alignment.")
        + legacy_table
        + paragraph("The released eager NeRD adapter takes 8.78 ms at batch 2,048 on this workstation. "
                    "The compact SuperNeRD eager model takes about 0.18 ms in the local forward harness. "
                    "The two figures cover different adapter implementations and model sizes. The "
                    "same-mass accuracy gap prevents treating the faster model as an equivalent solver. "
                    '<a href="results/nerd-throughput.json">Released adapter receipt</a> · '
                    '<a href="results/forward-timing.json">SuperNeRD timing receipt.</a>')
    )



def run() -> None:
    pilot = read("cartpole.json")
    structured = read("structured.json")
    rollouts = rollout_data(pilot, structured)
    extended = read("cartpole-extended.json")["policy"]["arms"]
    figure_data = {
        "policy_arms": pilot["policy"]["arms"],
        "extended_policy_arms": {"pathwise": pilot["policy"]["arms"]["pathwise"], **extended},
        "rollout_series": rollouts,
    }
    (RESULTS / "figure-data.json").write_text(json.dumps(figure_data, indent=2) + "\n")
    context = {
        "release": "v0.1",
        "date": "24 November 2026",
        "abstract": (
            "We study whether differentiating through a learned transition reduces the data and time needed "
            "to train a robot controller. We train a compact world model on Newton CartPole transitions "
            "and compare direct policy gradients with PPO on both the learned model and Newton. Across "
            "three training seeds, backpropagation reaches at least 90% success on 256 held-out Newton "
            "resets after 73,728 imagined transitions and reaches 100% at the fixed 294,912-transition "
            "budget. With more training, both PPO arms also reach 100%. Backpropagation reaches the 90% "
            "threshold 5.90× faster than PPO on the same learned simulator in policy training time and "
            "2.39× faster when including commissioning and scheduled evaluation. At batch 2,048 the learned graph advances "
            "a control step 8.77× faster than the local Newton graph, although unrestricted predictions "
            "drift over long open-loop rollouts. A CartPole-specific mechanical prior improves those "
            "rollouts. A released NeRD checkpoint is evaluated as a transfer baseline and FlashNeRD "
            "benchmarks are audited separately. The result establishes a policy-learning advantage on "
            "this restricted task under the measured recipes."
        ),
        "methods": Markup(methods()),
        "setup": Markup(setup(pilot)),
        "results": Markup(results_text(pilot, rollouts)),
        "released_baseline": Markup(release_text()),
        "limitations": Markup("""
<p>CartPole has two generalized coordinates, bounded forces and a rail joint limit. The policy starts
near upright. No floating base, variable contact set or dynamic-object interaction is exercised.
The unrestricted MLP's long-horizon error limits replacement of general articulated dynamics.
Mechanical structure helps CartPole because its equations are available and its teacher closely matches them.</p>
<p>The policy optimizer comparison uses one hyperparameter recipe per method and three seeds.
Evaluation resets are shared across seeds and methods. Episode counts therefore describe evaluation
coverage. The independent training seeds are the units of optimizer variation. Timing on one workstation
with warmed kernels does not establish performance on the GB200 or at FlashNeRD's largest batches.</p>
<p>The main NeRD transfer experiment changes the cart and pole masses, alongside the simulator implementation. Its eager
adapter and local CUDA graph timings are recorded separately. The latest benchmark's original data,
weights and optimized kernels are needed for a matched state-of-the-art comparison.</p>
"""),
        "conclusion": (
            "Short-horizon pathwise policy updates exploit a useful control gradient even when the learned "
            "simulator accumulates substantial long-horizon state error. The evidence motivates a matched "
            "ANYmal or G1 experiment that holds the contact representation and world model fixed while "
            "changing the optimizer. Comparable analytical-simulator policy quality is the deciding metric."
        ),
        "reproduce": Markup("""
<p>The repository pins Newton, Warp, the public NeRD asset and checkpoint hashes. It includes the run
commands, policy learning curves, model and policy weights, per-cohort prediction metrics and the source
for the figures. <a href="README.md">Reproduction instructions</a> ·
<a href="paper/replication-protocol.md">Protocol and follow-up targets</a> ·
<a href="results/cartpole.json">Raw pilot data</a></p>
"""),
    }
    env = Environment(loader=FileSystemLoader(ROOT / "paper"),
                      autoescape=select_autoescape(("html", "j2")))
    rendered = env.get_template("index.html.j2").render(context)
    clean = "\n".join(line.rstrip() for line in rendered.splitlines())
    (ROOT / "index.html").write_text(clean + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
