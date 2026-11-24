# Evaluate a deformable model on actions

For each case your model chooses one candidate. The scorer retrieves the result of executing that exact candidate in Newton. The same command works for both tasks with Python 3.10 or later and no external packages.

~~~sh
python tasks/evaluate.py cable --list > cable-inputs.json
python tasks/evaluate.py cable --score cable-choices.json
python tasks/evaluate.py cloth --list > cloth-inputs.json
python tasks/evaluate.py cloth --score cloth-choices.json
~~~

Supply a JSON object mapping every listed case ID to one integer, the zero-based index in its command list. Cable IDs combine geometry seed and target index, such as `"51240/0"`. A cloth ID is its fabric seed, such as `"50300"`. The listing supplies the goal and action programs. The per-task generators recover the full starting geometry and material parameters from the seed. Results include every executed choice, the number of successes and mean worst-gate or fold-shape error. Timing your planner and generating fresh confirmation cases require the per-task harnesses.

The two tasks use different physics and actuators. On the cable, two bounded load pulses must steer the tip to a pair of timed gates. On the cloth, the command gives the landing, lift and release-height parameters of a two-corner grasp program. Both use the same interface for choosing an action and verifying the outcome. All stored Newton outcomes are public; these cases are a reproducible benchmark for comparisons. Use new seeds for a blind evaluation.

[Cable programs, tolerances and released models](cable/README.md) · [Cloth programs, tolerances and folding model](cloth/README.md)
