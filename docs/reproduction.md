# Reproduction guide

## Installation and paths

Run the README's CPU quickstart first. It is a complete tested saved-result path.
The source checkout contains recipes and compact arrays; the installed package
contains code and the default asset manifest. From another directory pass the
absolute `--config`, `--data-root`, `--artifact-root` and `--output` you want.
`config:filename` resolves relative to its YAML. User-specified ordinary file
paths resolve relative to the calling directory. No lab mount or host is needed.

For model execution, use the separate family environment declarations rather than
installing NumPy 2 into the old simulator environments. See [installation](installation.md) and [testing](testing.md) for the installed
versions and actual short model checks. ManiSkill uses the corrected state-only
ManiSkill2 0.5.3/SAPIEN2.2.2 adapter; native Windows rendering is not a supported
execution path.

## Latent TD-MPC2: one fresh Stage-1 run

The following commands do actual scientific work after asset preparation. The complete target search is a stochastic paper workload; short prefix
updates and fixed-target integrations are tested separately. First bind the
recorded first common **development** order; do not use the held-out default for
target selection. This CPU helper copies existing panels/seeds without drawing
new ones and binds the task's recorded selection alpha:

```bash
python scripts/bind_order.py --config configs/tdmpc2/humanoid-walk.yaml \
  --order dev1 --run-index 1 --output outputs/hw-dev1.yaml
```

```bash
swaap prepare --config outputs/hw-dev1.yaml --seed 17 --output outputs/hw-data
swaap stage1 --config outputs/hw-dev1.yaml --seed 17 \
  --set assets.clean=outputs/hw-data/clean.pt \
  --set assets.reference=outputs/hw-data/reference.pt --output outputs/hw-targets
swaap select --config outputs/hw-dev1.yaml \
  --seed 17 --set assets.stage1_results=outputs/hw-targets/stage1.json --output outputs/hw-candidates
```

Selection preserves the Direct-return and deviation gates and exact Pareto/maximin
priority. A rejected run stays rejected; do not discard it from acceptance counts.
For each selected target, run construction and the same first development order
and 100 evaluation roots as every other candidate. Bind the target explicitly:

```bash
swaap construct --config outputs/hw-dev1.yaml \
  --set assets.target=outputs/hw-targets/targets/target_0100.pt \
  --set assets.clean=outputs/hw-data/clean.pt \
  --set assets.reference=outputs/hw-data/reference.pt \
  --alpha .9 --poison-fraction .1 --output outputs/hw-poison
swaap fit --config outputs/hw-dev1.yaml \
  --set method=clean --set assets.clean=outputs/hw-data/clean.pt \
  --set assets.reference=outputs/hw-data/reference.pt --output outputs/hw-clean-fit
swaap fit --config outputs/hw-dev1.yaml \
  --set assets.clean=outputs/hw-data/clean.pt \
  --set assets.reference=outputs/hw-data/reference.pt \
  --set assets.data=outputs/hw-poison/poison.pt \
  --set assets.order=outputs/hw-clean-fit/order.pt --output outputs/hw-poison-fit
swaap evaluate --config outputs/hw-dev1.yaml --set method=clean \
  --set assets.clean=outputs/hw-data/clean.pt \
  --set assets.reference=outputs/hw-data/reference.pt \
  --set assets.checkpoint=outputs/hw-clean-fit/fitted.pt --output outputs/hw-clean-eval
swaap evaluate --config outputs/hw-dev1.yaml \
  --set assets.clean=outputs/hw-data/clean.pt \
  --set assets.reference=outputs/hw-data/reference.pt \
  --set assets.checkpoint=outputs/hw-poison-fit/fitted.pt --output outputs/hw-poison-eval
```

The iteration path above illustrates binding a **selected** target; it is not a
promise that iteration100 passes selection in a new seed. Supply the actual path
from `select.json`. For artifact reproduction use the named published target ID
instead. `scripts/run_attack.py` combines these fixed-target stages serially.
Stage1 retains its own per-iteration trajectory collection; the prepared fixed
banks supply Stage2 and victim fitting, rather than replacing Stage1 collection.

Development-selection JSON contains one row per candidate/order:
`candidate_id`, `checkpoint_sha256`, `frozen_rank`, `fit_order`,
`matched_clean:true`, `attack_returns:[...]`, `clean_returns:[...]`.
The arrays are the corresponding `evaluate.json` episode returns, without
averaging them into independent attacks. Pass that explicit JSON as
`assets.development_results` to `swaap select --set selection.phase=screen`.
Only the top three receive the other two common development orders. Supply their
three-order rows to `selection.phase=winner`. The frozen winner is evaluated on
three new held-out orders/root panels; never use final returns to change it.
Use `bind_order.py --order dev2`/`dev3`, then `hold1`/`hold2`/`hold3` for the
corresponding common schedules; the helper changes the final construction alpha
only after entering a held-out role. All81 published held-out panel/fit-seed joins
were checked against the original winner manifest before exporting these panels.
Keep the same newly prepared Clean/reference bank overrides in every subsequent
bound configuration; do not fall back to the published banks midway through a
fresh experiment.
Four tasks have target-selection alpha .5 and final alpha .9: their separate
`*_selection.yaml` recipes preserve that distinction.

An aggregation JSON contains `task`, `method`, `stage1_run`, `fit_order`, and
`returns` for each final fitting order. Run `swaap aggregate --config ...
--set assets.results=your-final-rows.json --output outputs/summary`. Repeat Stage 1
with independently declared seeds for independent replications; three fitting
orders of one target do not constitute three independent Stage-1 runs.

## Other families and ablations

`docs/paper_coverage.md` maps retained paper labels to concrete recipes. The same
stage commands call explicit latent, gray-box or DINO implementations; their
observation domains, editable fields, fitting reductions and precision differ.
See `docs/families.md` before gray-box/DINO execution. Do not substitute the
global-support DINO controls or developmental ManiSkill DIRMAG routes for the
retained experiment.

## Incoming-buffer CUSUM

The detector operates **before fitting**, using raw Original-model residual L2
sequences. It is not the post-adaptation rollout diagnostic. A residual NPZ has
numeric `residuals[trajectory,step]`, non-object `trajectory_ids`, and a scalar
Unicode `original_sha256`. Include all incoming trajectories, even those with no
modified row. Use genuinely disjoint roots for statistics, threshold calibration, independent
Clean test and incoming data. Role-independent trajectory identities detect
reused roots; changing a role label does not create independence.

```bash
swaap calibrate-detector --statistics statistics.npz --threshold thresholds.npz \
  --family maniskill --output detector.json
swaap score-detector --calibration detector.json --clean-test independent-clean.npz \
  --poison incoming-poison.npz --output incoming-metrics.json
```

The unchanged recurrence uses median/MAD, drift6, split-conformal rank
`ceil((n+1)*(1-FPR))` and strict `score > threshold`. The calibration is frozen
before poison scoring. The [detector guide](detectors.md) gives exact task-specific counts and complete
fresh Original-policy collection commands, including gray-box and DINO-WM
adapters. All scores, AUC, standardized pAUC and Wilson/bootstrap intervals retain
trajectory denominators. Collection, threshold freezing and scoring are explicit
commands; scoring never silently collects additional episodes.

## Cost and reproducibility

For Original-reference rollout error, retain the exact matched Clean arrays. An
episode manifest names `task`, `stage1_run`, `fit_order`, `episode_index`,
`environment_seed`, `planner_seed`, `expected_steps`, `arrays`/`arrays_sha256`,
`original_checkpoint_sha256`, `clean_fitted_checkpoint_sha256`,
`predictor_role: Original` and `rollout_role: matched Clean` for every episode.
If `original_predicted_next_latents` already survives in the NPZ, aggregate it
without loading a model:

```bash
swaap aggregate --config configs/tdmpc2/humanoid-walk.yaml \
  --set aggregation.metric_mode=original_on_matched_clean \
  --set assets.saved_rollouts=clean-reference-manifest.json --output outputs/clean-reference
```

If those predictions were not retained, the command explicitly reports a missing
measurement. The separate `swaap evaluate --set
evaluation.mode=saved_original_predictions` producer (with the same manifest,
Original and task recipe) performs frozen-model inference on the saved Clean
transitions and creates new arrays. It runs no environment, fitting or poisoning.
A frozen-Original forward on a saved500-transition matched-Clean H-Walk
trajectory and its CPU-only aggregation have been checked on the real model.
This is a producer test, not a rerun of every paper reference measurement.
Own-model predictions or Original's different rollouts cannot substitute for
this reference.

The nine-task primary/comparison/TRIM allocations total275.57 GPU-hours; the grid
added169.63 GPU-hours for471 new conditions with27 reused. These are bounded
campaign totals, not per-attack latency or a full cost for every paper family.
Existing receipts measure one H-Walk Stage1 at101.08 minutes (GPU not recorded),
an A100 200-update fitting timer around1.31 seconds, and episode-clock totals
around315.13 seconds per100 H-Walk episodes. Those timers have different loading,
checkpointing and reset boundaries. There is no isolated nine-task Stage2 timing;
none is inferred by division or a new benchmark.

New stochastic returns can differ by device, physics runtime or random streams.
Exact saved-result reproduction, small numerical parity tests and full fresh RL
reproduction are distinct levels of validation.
