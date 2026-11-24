# Folding with a training-range check

The optional planner uses the learned model to shortlist candidate grasps when all fabric and command descriptors fall inside the observed training ranges. Otherwise it immediately evaluates the complete candidate bank in Newton. This preserves a physical action-selection path when extrapolating its learned shape would be unreliable.

| Planner on the 16 original confirmation cases | Successful folds | Actual Newton episodes | Mean final shape error (% of length) |
| --- | ---: | ---: | ---: |
| Frozen model shortlists three | 14/16 | 48 | 2.027 |
| Training-range check, then three or all twelve | 14/16 | 66 | 2.027 |
| Newton evaluates all twelve | 15/16 | 192 | 2.001 |

The guard abstained on two fabrics. Seed 50301 fell outside the observed density range. Seed 50315 fell outside the membrane-stiffness range. Both completed a fold after twelve native evaluations. It accepted 50300 and 50302–50314 and checked three actions on each. The failures were 50300 and 50314. The guard adds 18 native episodes relative to the three-check planner, and avoids 126 relative to exhaustive search. Every newly simulated outcome exactly matched the stored candidate measurements. The 16 original confirmation cases were known before this safety extension. This is a retrospective comparison. [New native replays and decisions](guarded-confirmation.json) · [Original comparison](confirmation.json)

## How invocation works

The checkpoint stores 200 normalized training inputs. The code reads the extrema of all five fabric descriptors over all 200 inputs and the extrema of the three moving-action descriptors over the 192 moving inputs. Eight stationary examples form a separate exact command. Limits come exclusively from the recorded checkpoint. The comparisons use its original float32 scaling and inclusive bounds. If any proposed moving command is outside bounds, the whole bank falls back so a rejected command can still win in Newton. A valid, in-range query evaluates up to three candidates. A rejected query skips loading or running the neural network and evaluates every candidate supplied by the caller. The caller retains the full physics scorer and real actuators or their chosen teacher approximation. [Calibration rule](guard-protocol.txt) · [Optional planner](guarded_planner.py)

Axis-wise ranges guard against extrapolation in individual recorded descriptors. They cannot establish that every combination was covered, that the predicted mesh obeys contact, or that the ranker will find a successful action. Use the entry point for the fixed topology, control schedule, starting family and material parameterization of the public benchmark. Changes invisible to the eight descriptors require separate validation.

~~~sh
python guarded_planner.py --calibration
python guarded_planner.py --seed 50301
python guarded_planner.py --confirm --workers 5 --output new-confirmation.json
python -m unittest test_guarded_planner -v
~~~

For an existing Newton task, instantiate GuardedFoldPlanner(checkpoint) once, then call its select(geometry, actions, target_landing, native_score) method. The callback receives a candidate index and returns the task cost after physical execution. The returned record contains the decision, the reasons for fallback, the actual checked indices and the winning index. [Branch and boundary checks](test_guarded_planner.py)
