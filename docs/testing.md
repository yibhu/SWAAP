# Verification and its limits

The source includes CPU scientific-parity tests and explicit reduced real-model
recipes. Importing a module, parsing a configuration and replaying a saved table
are different checks from running a simulator and model.

## CPU checks

```bash
python -m pip install -r environments/cpu-validated.txt
python -m pip install --no-deps .
python -m pytest -q
python scripts/check_recipes.py
```

Use this CPU environment separately from simulator environments. The tests compare
pinned definitions on identical small inputs: Stage1 barrier and gradient/update
behavior, Stage2 losses/gradients/projection, top-k and frozen-field behavior,
exact tied-field gradients, gray-box encoder separation, DINO history/action masks,
precision boundaries, detector statistics/scoring, and nested aggregation. The
code-only source snapshot omits numerical assets; install `saved-results` and set
`SWAAP_SAVED_DATA=artifacts/results/saved` to include all144 main-table assertions.
Missing assets are not replaced by synthetic paper values.

The ordinary fresh-run script has CPU handoff tests using the real admission,
screening and winner selectors with synthetic model-stage fixtures. They check
rejection preservation and winner locking before held-out evaluation. A complete
fresh target search/race has not been rerun for release engineering.

## Real-model integration

All model execution below used clean exported source installed as a package and
explicit public-format assets, outside the research checkout. Each endpoint was
saved and reloaded before its complete declared evaluation episode or goal.

| Execution family | Hardware/software tested | Reduced work actually completed |
|---|---|---|
| DMControl H-Walk | RTX3090, Linux/WSL, frozen Torch2.6 CUDA12.4 and MuJoCo3.1.2 | One Stage1 outer update; two SWAAP updates; matched ten-epoch/200-update fits; one complete500-decision episode per endpoint; two Original detector trajectories |
| MetaWorld Soccer | RTX3090, Linux/WSL, exact legacy MetaWorld commit and MuJoCo210 | One Stage1 outer update; two SWAAP updates; matched20-update fits; one complete100-decision episode per endpoint; two Original detector trajectories |
| MyoSuite Reach | RTX3090, Linux/WSL, MyoSuite2.10.3/MuJoCo3.3.0 | One Stage1 outer update; two SWAAP updates; matched20-update fits; one complete100-decision episode per endpoint; two Original detector trajectories |
| Corrected ManiSkill Lift | RTX3090, Linux/WSL, source-bound0.5.3/SAPIEN2.2.2 state-only adapter | One Stage1 outer update; two SWAAP updates; matched20-update fits; one complete100-outer/200-native-step episode per endpoint; two Original detector trajectories; zero renderer/Vulkan-entry calls |
| Raw gray-box Soccer→MT80 | RTX3090, same isolated legacy simulator | Released target; two raw constructor updates; matched two-update fits; one complete100-decision episode per endpoint; one source-policy detector trajectory scored by the distinct MT80 reference |
| DINO-WM PushT | RTX5080, Linux/WSL, exact Torch/vision CUDA12.8 wheels | One Stage1 outer update; two constructor updates; matched two-update fits; one complete planning goal per endpoint; one genuine ten-action Original collection trajectory/eight detector windows |

The corrected ManiSkill adapter also passed CPU source/reset/state-only guards
for all three tasks. This is not a claim of native-Windows rendering support or
bitwise native-3090A/A100 parity for the newer corrected source. The old Gym-based
ManiSkill qualification does not replace the corrected Gymnasium checks.

Short Stage1 prefixes are **not qualifying attack targets**. Their preserved
outcomes are separate from the released targets used for integration. Negative
or weak smoke effects were retained; no seed/target was changed to produce harm.

The gray-box/DINO released detector cohorts were also calibrated and scored
through the installed package. The same frozen DINO calibration scores both
faithful poisons. These CPU replays use existing residuals and do not collect a
new800-trajectory DINO panel. A frozen-Original forward on a saved H-Walk Clean
trajectory and its separate CPU aggregation test the Original-reference producer
without adding environment episodes.

## What these tests establish

These are engineering checks of source extraction, real dependencies, artifact
handoffs, masks, horizons, save/reload and explicit outputs. They do not establish
paper efficacy from one episode, rerun all tasks/conditions, or certify identical
RL returns across devices. Full alpha/fraction searches, three-run experiments
and complete detector cohort collection are the separately documented scientific
workloads. Existing paper results remain bound to their original saved inputs.

The package uses the recorded scientific versions. The resolved dependency files
and verified wheel bodies describe the tested installation; they do not claim
bitwise identity for every unrecorded historical system library. Startup failures
were preserved in the engineering report and included in its resource account.
No paper measurement was silently replaced by a smoke output.
