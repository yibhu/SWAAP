# TD-MPC2 and ManiSkill reproduction

The public latent workflow uses the executed SWAAP implementations, including
TD-MPC2's model, actor and planner support. ManiSkill changes the environment
adapter and its recorded fitting schedule. It does not use another attack
controller. No command starts a scheduler, monitor, notebook, or remote worker.

## Validation scope

CPU tests compare exported losses, gradients, top-k indices, projections and
updates with the pinned scientific functions. They also cover the Stage-1
barrier, detached running-scale snapshots, policy binding, fresh-Adam fitting,
structural checkpoint metadata, and incoming-buffer detector handoffs.

A clean installed export has run the H-Walk fixed-target example on an RTX3090
under Ubuntu22.04/WSL: two construction updates, matched E10/B256/200-update
fits, checkpoint save/reload, and one complete 500-decision episode per endpoint.
The real Stage-1 outer update and two Original-policy detector trajectories have
also executed. These are short integration checks, not paper efficacy reruns.
The same bounded real-model check has passed on Myo-Reach and rendererless
Lift-Cube with one native Stage-1 outer update, two constructor updates, matched
20-update fitting, checkpoint save/reload, full endpoint episodes and Original
detector samples. The MetaWorld/Soccer representative passed the same real-model integration
with its separate source-pinned legacy environment.
These checks establish executable plumbing; they are not a rerun of the paper.
CPU imports alone are not treated as simulator validation.

The scientific core uses Torch `2.6.0.dev20241112+cu124`. Its actual historical
wheel body has been downloaded, hashed, inspected and installed with the
historical Triton dependency on an isolated environment. No modern Torch is
silently substituted. The resolved dependency lists in `environments/` describe
that engineering installation; they do not assert that every historical
transitive dependency had identical bytes. The explicit
`runtime.allow_unqualified_software=true` override labels a different software
substrate and is not used for the validated example.

## Installation and assets

Use a separate environment from saved-table reproduction. The family YAML files
pin the core dependencies. The `*-resolved.txt` files additionally pin the
packages installed during validation. For DMControl, from the source root:

```bash
conda create -n swaap-dmc -c conda-forge python=3.9.21 pip=24.0
conda activate swaap-dmc
python -m pip install --extra-index-url https://download.pytorch.org/whl/nightly/cu124 \
  -r environments/tdmpc2-resolved.txt \
  'torch @ https://download.pytorch.org/whl/nightly/cu124/torch-2.6.0.dev20241112%2Bcu124-cp39-cp39-linux_x86_64.whl#sha256=387a1fc1e164c97c149c868a276ce205b8a2ffa5b59209747e5bcc8c103533fb'
python -m pip install --no-deps .
python -m pip check
```

Use the matching MyoSuite or ManiSkill resolved file in a separate environment.
TorchRL is not imported by this code and is not installed as an unused optional
dependency. Family requirements are:

- DMControl: the pinned `dm-control`/MuJoCo packages in the declaration. Custom
  DMControl task code and XML are included with the retained hashes.
- MetaWorld: legacy commit
  `04be337a12305e393c0caf0cbf5ec7755c7c8feb`, legacy Gym and `mujoco_py`/MuJoCo
  2.1 are required. This is not today's MetaWorld API. See the gray-box guide for
  the same legacy simulator installation boundary. The original wrapper method
  bodies are statically bound to the legacy-Gym wrapper base, as in execution.
- MyoSuite: use `environments/tdmpc2-myosuite.yml`. The retained qualification
  binds MyoSuite `2.10.3`, MuJoCo `3.3.0`, NumPy `1.23.5`, legacy Gym `0.23.1`
  and Torch `2.6.0.dev20241112`; no Git commit is asserted for that installed
  package. `environments/MYOSUITE_PROFILE_PROVENANCE.json` binds the receipt
  SHA and separates measured versions from package dependency constraints.
  Reach and Pen use `envs/myo/assets/hand/myohand_pose.xml` and
  `myohand_pen.xml`, including the package's `simhive/myo_sim` hand/scene assets
  (recorded VERSION `0.1.0`). These hand XML and myo_sim assets are included
  in the installed `myosuite==2.10.3` distribution; the validated install needed
  no private asset alias, external research checkout or additional download.
  The optional separately licensed `myo_model`
  download is not referenced by these two task definitions; do not automatically
  accept its license. The historical qualification covered dependency/input/reset
  identity only. The new isolated public Myo-Reach integration additionally
  passed a real Stage-1 outer update, two Stage-2 updates, matched fitting and
  checkpoint save/reload, full 100-step endpoint evaluations, and Original-policy
  detector collection. Package and asset licenses remain separately applicable.
- ManiSkill: `tdmpc2-maniskill.yml` and `tdmpc2-maniskill-resolved.txt`
  bind the corrected Gymnasium source to ManiSkill2 `0.5.3`, SAPIEN `2.2.2`,
  NumPy `1.23.5`. The official wheel matches the adopted saved simulator-source
  hashes. The earlier Gym-based `0.4.1` environment is a different source and
  its old qualification does not establish parity for this adapter. The tracked
  state-only subclasses verify seven upstream source hashes and inherit native
  task/control/reset/step/reward methods unchanged, while omitting visual-only
  construction. They do not rewrite installed modules or create a renderer.
  Rendering/camera entry is prohibited. SAPIEN still dynamically links the
  system `libvulkan1` loader even when it makes no Vulkan calls; on Ubuntu22.04
  install that runtime library through the normal system package manager.
  State observations, `pd_ee_delta_pos`, `normalized_dense`, exactly 100 outer
  decisions and two native steps per decision are explicit. Both rewards are
  accumulated; native success/termination does not shorten this horizon.
  Original checkpoint numbers are Lift 1, Pick 1 and Stack 3.

Use `swaap assets list` and the asset guide to obtain/import verified Originals
and clean/reference banks. Asset-based evaluation also needs the exact selected
Stage-1 target, poison or fitted dynamics checkpoint. Large models and banks are
not committed in the source tree. The asset bundle guide provides acquisition/import and checksum commands for
the exact inputs. A new data collection or target optimization is a separate
fresh experiment, rather than a substitute for a published artifact.

The recipes reference named asset IDs. Explicit local inputs use
`--set assets.NAME=/absolute/path/file`. Root panels shipped beside a recipe use
`config:FILENAME` and are resolved relative to that recipe, independent of the
current working directory. Only import trusted `.pt` files: they are the original
Torch artifact format, not an interchange format for untrusted arbitrary files.

## One working quickstart

From a clean install with the CPU test dependency:

```bash
swaap smoke --output outputs/synthetic-smoke
python -m pytest tests/test_latent_science.py tests/test_latent_extensions.py
```

The smoke uses the actual ordinary Stage-2 loss and projected updates on tiny
synthetic CPU tensors. It writes a synthetic poison and a PASS record. It loads
no pretrained model and makes no environment episode. This is not an attack
against a paper checkpoint.

## Artifact-based fitting and evaluation

The following commands perform real model work and require the corresponding
family environment and exact assets. Output directories must be new. Nothing
is executed by installation or `--help`.

```bash
# Fit the exact imported poisoned bank from Original using the saved order.
swaap fit --config configs/tdmpc2/humanoid-walk.yaml \
  --set assets.data=/absolute/path/poison.pt \
  --set assets.order=/absolute/path/order.pt \
  --set fit_seed=42801 --output outputs/hwalk-fit

# Evaluate that endpoint on the common root panel in the configuration.
swaap evaluate --config configs/tdmpc2/humanoid-walk.yaml \
  --set assets.checkpoint=outputs/hwalk-fit/fitted.pt \
  --output outputs/hwalk-evaluation

# Direct Model Poisoning deploys target dynamics; it performs no victim fit.
swaap evaluate --config configs/tdmpc2/humanoid-walk.yaml \
  --set method=direct --set assets.target=/absolute/path/target.pt \
  --output outputs/hwalk-direct
```

The first task recipe includes the first published contributing run's final root
panel and fit seed. `configs/tdmpc2/published_orders.json` contains all 27 runs'
exact target identities, three order seeds and 100-root panels.
`configs/maniskill/published_orders.json` keeps all nine attempted slots, including
the Lift run with no qualified final candidate. Only two Lift runs and three
Pick/Stack runs contribute numerical main results. No missing outcome is zero.
Use the recorded order seed matching the imported `order.pt`; the example seed
must not be borrowed for another run.

The same index binds each published SWAAP(Random) construction seed separately
from the fitting seed. `scripts/bind_order.py` exports those bindings. The
constructor requires `legacy_random_seed` or an explicit `construction_seed`;
it never silently replaces a missing run-specific random seed with zero.
Random500 and learned-target/random-support controls remain separate methods.

To evaluate Original explicitly, use `--set method=original`. All fitted arms
require `assets.checkpoint`; an absent checkpoint cannot silently evaluate
Original and label it SWAAP. Clean means the matched Clean-fitted endpoint, not
necessarily Original.

## Constructing a fresh attack

Each stage is an ordinary command and reads the artifacts produced by the
preceding stage. No hidden research-controller state is needed.

1. `swaap prepare`: collect the five-column clean/reference banks from Original,
   or import the exact published banks instead. ManiSkill's retained collection
   uses 50 real Clean trajectories without action replacement and 500 imagined
   Original-reference trajectories with 3% random actions, each starting from an
   actual reset. Its collection root panels are shipped. Nine-task fresh
   collection exposes the retained collector's explicit settings; exact published
   bank reproduction should use the verified bank artifacts until their complete
   original collection provenance has been turned into an installation-tested
   input recipe. New collection is not saved-result replay.
2. `swaap stage1`: one complete task-budget run, saving every tenth dynamics
   checkpoint and its ten-episode isolated Direct assessment. The Original
   reference is measured within that same run. No efficacy early stopping or
   automatic extra attempt is implemented.
3. `swaap select --set selection.phase=targets
   --set assets.stage1_results=outputs/stage1/stage1.json`: apply the frozen
   Direct≤0.9 Original/current-relative-deviation≤0.30 criterion, require five
   distinct eligible checkpoints and rank at most 20 by native Pareto/maximin.
   Nothing downstream enters this ranking.
4. For every selected target, run `swaap construct` with `assets.target`, then
   matched `fit` and `evaluate` commands on the common development order. Use
   `selection.construction_alpha`, not the adopted main-table alpha, while
   selecting targets. The four task-specific `.5` selection recipes are separate
   `*_selection.yaml` files. The `.9` main recipes reuse the selected target;
   they do not repeat selection at `.9`.
5. Supply the paired development vectors to `swaap select` with phase `screen`.
   It retains the top three by harm in return units. Give those candidates two
   further common development orders, then call phase `winner`. It selects by
   three-order mean, frozen rank and checkpoint-SHA ties.
6. Construct the final operating point on the locked target, then use three new
   fitting orders and disjoint 100-root evaluation panels. Reuse the same order
   and roots for matched Clean and every comparison arm.
7. `swaap aggregate --set assets.results=/path/results.json`: average episode
   returns within each fitting order, then fitting-order means within each
   independent Stage-1 run, then report mean/sample SD across those runs.

Selection JSON contains `candidate_id`, `checkpoint_sha256`, `frozen_rank`,
`fit_order`, `matched_clean: true`, `clean_returns`, and `attack_returns` for each
candidate/order. Phase `screen` requires one common order; `winner` requires
three. ManiSkill additionally requires frozen `delta_d`, `delta_d_ref`, `tpr5`
and `clean_fpr5` per poison and applies its 1.25× deviation and +.05 alarm gates
before return ranking. The three inherited ManiSkill targets retain their
actual development-selected history; a new run is not described as another
independent validation of those selections.

The default Stage-1 roots in a fresh recipe are an explicitly fixed ten-root
panel; the full published final panels are separately preserved. Change task by
choosing its recipe. Use `--seed`, `--alpha`, `--poison-fraction`, and `--set`
for documented new configurations without editing scientific source. To change
victim permutations use `--set fit_seed=INTEGER`; a published saved order's seed
is checked and cannot be relabeled by changing only `--seed`.

## Retained comparison and ablation commands

The same `construct`, `fit` and `evaluate` stages keep these arms distinct:

| Method/configuration | Execution and required input |
|---|---|
| `clean` | Copy Clean bank; fit Original dynamics with matched permutations. |
| `direct` | Evaluate a saved Stage-1 target, no victim fit. |
| `raw_dmp` | Copy target successor predictions onto the native top-residual support; no Stage-2 refinement. |
| `swaap` | Clean-label initialization, native target-residual support, cosine matching plus original selected-row regularizer. |
| `swaap_legacy_random` | GPU-global multiplicative target draw followed by NumPy Bernoulli support; this is main-table SWAAP (Random). Realized count is retained. |
| `random500` | Exact-count random support, isolated CPU multiplicative target and step size 1; separate control. |
| `swaap_random_support` | Learned target, random exact-count support, ordinary step size 10. |
| `random_target_shared_support` | Exact saved SWAAP support, prescribed random target tensor/seed and step size 10. This is the step-matched comparison, not Random500. |
| `swaap_fixed_support` | Ordinary constructor on explicitly supplied support, including reset-safe matched controls. |
| `trajectory_consistent` | Earlier sequential diagnostic-loss surrogate and detached two-field writes. It is not the exact-tied objective. |
| `exact_tied_trajectory` | One differentiable latent in both stored fields; matching from actual full-buffer victim MSE. |
| `ordinary_seeded_tied` | Same exact-tied loss, initialized from a completed ordinary poison on the same support. |
| `*_trim.yaml` | `fit.defense=trim`: corrected matched-epoch TRIM, beta .2, 160 updates; apply separately to Clean, SWAAP and raw DMP data. |
| `mw-push_online_pgd_*.yaml` | Evaluation-only exact iterative online PGD at .01/.02 on an existing matched Clean checkpoint. Offline .015/.05 is not substituted. |
| `*_fixed500_buffer*.yaml` | `prepare` appends certified extra Clean rows behind the unchanged original 5k bank and 500-row poison. Requires `assets.extra_clean`, `poison`, `support`. Refit with the declared bank size; no reconstruction. |
| `*_fixed_ratio_buffer25000.yaml` | Full-bank residual support selection at rp .1, 2,500 edited labels, E10/980 updates. |

The alpha/rp and MPC menus are compact declarations in
`sensitivity_grid.yaml` and `mpc_grid.yaml`, not executable campaigns. Run one
explicit task recipe/condition at a time with the ordinary CLI. For example,
`--alpha .5 --poison-fraction .2` changes the constructor only; it does not select
a new target. MPC evaluation overrides `model.num_samples` and `model.horizon`
before constructing the retained planner; it reuses the exact saved model and
performs zero fitting. CUSUM has separate incoming-buffer calibration/scoring
commands; rollout errors are never presented as incoming-buffer detection.

## Numerical behavior and file layout

`src/swaap/latent/SOURCE_PROVENANCE.json` records original executed hashes,
public paths and mechanical extraction changes. The Stage-1 effective path uses
the joint squared-gradient barrier denominator; stateful Q-scale updates with a
detached cloned denominator at each call; the current outer policy in both
transition-gradient estimates; separate clipped Adam updates to dynamics and
surrogate policies; no critic/encoder/reward/termination weight update. Stage 2
retains native Torch top-k ties, full-buffer mean MSE, unregularized cosine
normalization (no invented epsilon), selected-row mean regularization, step-10
updates and clean-centered L-infinity projection followed by [0,1] clipping.
Original components and unsupported buffer fields are checked for equality.

A run directory contains the resolved configuration plus one stage summary.
Stage 1 adds `targets/target_XXXX.pt` and matching compact assessments.
Construction adds `poison.pt`, `support.pt`, and numerical history; fitting adds
`order.pt` if newly generated and `fitted.pt`; evaluation adds one episode array
per episode and the ordered return/deviation vectors. Partial output is never
automatically replayed or overwritten. No private per-job Python environment is
copied. Configure disposable compiler/cache state through the machine's usual
node-local `TMPDIR`/cache variables; persistent directories are for scientific
artifacts, not duplicated environments.

The installation and validation guides distinguish the bounded real-model tests
from a full scientific reproduction. Fresh RL results can vary across devices
and random seeds even when the saved numerical replay is exact.

## Original-reference error on matched Clean trajectories

The Clean-reference column is **Original's prediction error on the matched
Clean-fitted model's rollouts**, not either model's own-rollout error. Ordinary
`evaluate` saves realized latents, actions and the evaluated endpoint's
predictions; those endpoint predictions alone cannot fill that reference column.

An explicit inference-only producer is available for saved Clean rollouts:

```bash
swaap evaluate --config configs/tdmpc2/humanoid-walk.yaml \
  --set evaluation.mode=saved_original_predictions \
  --set assets.saved_rollouts=/path/clean-rollout-manifest.json \
  --output outputs/clean-original-predictions
swaap aggregate --config configs/tdmpc2/humanoid-walk.yaml \
  --set aggregation.metric_mode=original_on_matched_clean \
  --set assets.saved_rollouts=outputs/clean-original-predictions/saved_rollouts.json \
  --output outputs/clean-reference-metric
```

The first command loads frozen Original only and predicts once per saved episode
on its exact float32 latent/action rows. It runs no simulator, fitting, poison
construction or new episode, and writes new arrays without changing source
rollouts. It requires qualified model inference resources. The second command
is CPU-only and uses the unchanged adopted float64 `episode_delta_m` reduction.
If Original predictions already survive, run only the second command.

The manifest contains an `episodes` list. Each record specifies `task`,
`stage1_run`, `fit_order`, `episode_index`, `environment_seed`, `planner_seed`,
`expected_steps`, `original_checkpoint_sha256`,
`clean_fitted_checkpoint_sha256`, `rollout_role: "matched Clean"`, and `arrays`
(an NPZ path, relative to the manifest or absolute). CPU scoring additionally
requires `predictor_role: "Original"` and the array key
`original_predicted_next_latents` beside `latents`. The inference producer also
requires saved `observations` and `actions`. Optional `arrays_sha256` is checked.
An absent Original prediction produces an explicit missing-measurement error;
no fitted-model prediction is substituted.

## Incoming-buffer detector collection

`collect-detector` follows the frozen Original policy and predicts each complete
trajectory in one float32 batch. It records Original identity, environment/planner
seeds and role-independent trajectory IDs. Reusing roots under another role is
caught by the calibration/scoring overlap checks. It does not score adapted-model
rollouts as incoming data. ManiSkill uses the adopted CPU float32 residual norm;
the nine-task path uses the adopted native-device float32 norm.

For example, the exact published ManiSkill root lists make the full collection
commands explicit (these commands create new episodes):

```bash
swaap collect-detector --config configs/maniskill/lift-cube.yaml \
  --set detector.role=statistics \
  --set assets.roots=config:lift-cube_detector_stats_roots.json \
  --output outputs/lift-statistics
swaap collect-detector --config configs/maniskill/lift-cube.yaml \
  --set detector.role=threshold \
  --set assets.roots=config:lift-cube_detector_threshold_roots.json \
  --output outputs/lift-threshold
swaap calibrate-detector --family maniskill \
  --statistics outputs/lift-statistics/residuals.npz \
  --threshold outputs/lift-threshold/residuals.npz \
  --output outputs/lift-detector.json
swaap collect-detector --config configs/maniskill/lift-cube.yaml \
  --set detector.role=clean_test \
  --set assets.roots=config:lift-cube_detector_clean_test_roots.json \
  --output outputs/lift-clean-test
```

The supplied lists contain disjoint 500/500/1000 trajectories. Incoming poison
scoring additionally takes the exact source-bank trajectory roots, the bank,
poison, locked calibration and `assets.clean_test` residuals via `detect-buffer`. It scores all 50 ManiSkill
trajectories or ten DMControl trajectories, including unmodified trajectories.
`detect-buffer` also produces the calibrated metrics in `cusum.json`;
`score-detector` is the CPU-only route when residual arrays are already saved. Nominal and realized FPR are reported separately. A two-trajectory
`detector.role=smoke` collection only tests the interface and is not a detector
calibration. Saved-cohort scoring can use the verified residual bundles directly,
without recollection or model inference.

## Fresh bank provenance for detector scoring

For a new stochastic experiment, `scripts/make_detector_roots.py` declares
mutually disjoint statistics, threshold, independent-test, incoming-bank and
reference-bank roots without executing a model. For example:

```bash
python scripts/make_detector_roots.py --task humanoid-walk --seed 123 \
  --original tdmpc2.humanoid-walk.original --artifact-root artifacts \
  --output inputs/hwalk-roots
swaap prepare --config configs/tdmpc2/humanoid-walk.yaml \
  --set data.mode=explicit_roots \
  --set assets.collection_roots=inputs/hwalk-roots/collection.json \
  --output outputs/hwalk-data
```

This opt-in collection calls the unchanged native collector once per explicitly
seeded trajectory. It preserves the stated model, planner, action-replacement
probabilities and real-versus-imagined reference settings. Its RNG batching is
new; it does not claim byte-identical regeneration of historical banks. The
output `clean_trajectory_roots.json` binds the exact stored bank SHA, Original
SHA, roots and row ranges. Pass it as `assets.trajectory_roots` to
`detect-buffer`, along with the constructed poison and the already locked
calibration plus independent clean-test residuals. Default historical collection
and artifact imports keep their existing semantics. Unknown historical roots
are not fabricated by this mode.

## Recorded shared-support and trajectory comparisons

The random-target comparison binds the prescribed random-target seed and tensor
content hash separately from the construction seed. The order helper reads all
nine retained Push/H-Walk/Dog-Run bindings from `published_orders.json`; it never
creates a different target after observing returns. For example:

```bash
python scripts/bind_order.py \
  --config configs/tdmpc2/humanoid-walk_random_target_shared_support.yaml \
  --run-index 1 --order hold1 --output inputs/hwalk-random-target.yaml
swaap run --config inputs/hwalk-random-target.yaml \
  --output outputs/hwalk-random-target
```

The exact-tied comparison has five source-bound conditions in
`configs/tdmpc2/trajectory_conditions.json`: three H-Walk runs and two Dog-Run
runs. Their 250 ordered intermediate-state indices are compact JSON files,
verified against the original closed JSON/Torch support inputs. The loader
preserves int64 order and rejects duplicates or invalid indices; it never
reranks or repairs a mask. Target identities, alpha .9, poisoning fraction .05,
projection/update settings, episode boundaries and development exposure remain
explicit. The same helper supplies the three recorded fitting orders:

```bash
python scripts/bind_order.py \
  --config configs/tdmpc2/humanoid-walk_exact_tied_trajectory.yaml \
  --run-index 2 --order hold1 --output inputs/hwalk-tied-run2-hold1.yaml
swaap run --config inputs/hwalk-tied-run2-hold1.yaml \
  --output outputs/hwalk-tied-run2-hold1
```

Repeat that exact construction's fitting/evaluation on hold2 and hold3 using the
corresponding bound orders; do not construct a new poison per fitting order.
The constructor uses one differentiable state in both stored fields and matches
the full-buffer victim MSE. The older separate consistency-gradient surrogate
is not used for this table. Raw DMP-data artifact reproduction likewise supplies
the exact frozen `assets.support`; it does not reselect that support.

## One complete fresh task run

The ordinary serial driver assembles development handoffs automatically:

```bash
python scripts/run_fresh_task.py \
  --config configs/tdmpc2/humanoid-walk.yaml \
  --seed 123 --panel-position 1 --artifact-root artifacts \
  --output outputs/hwalk-fresh-seed123
```

By default it collects new banks, runs one complete Stage-1 budget, preserves all
saved candidates/rejections, applies the frozen admission and native K20 ranking,
constructs every selected candidate, screens on one common development order,
compares the top three on two additional orders, locks the winner, and performs
three disjoint final-order evaluations. Matched Clean fitting/evaluation is
reused within each order. The four task-specific target-selection alphas are
preserved separately from the final operating alpha. If they match, the saved
winning poison is reused; otherwise exactly one final construction is made on
the locked target. There is no retry, alternative target after final evaluation,
or automatic second Stage-1 attempt.

Use `--reuse-input-banks` to retain the verified published clean/reference banks
while optimizing a new target. This is an artifact-assisted fresh experiment,
not a reproduction of the historical selected target. The driver writes one
`workflow.json`, immutable stage products, compact development/final result
vectors and a nested aggregate. A rejected run is a final recorded outcome.
One run has one independent Stage-1 statistical unit; its three fitting orders
are not three independent attacks.

This driver covers the nine DMControl/MyoSuite/MetaWorld tasks. It explicitly
rejects ManiSkill, whose corrected development-selected targets and detector
gates remain in their family-specific recipes. The driver handoff has CPU tests
with real selectors and synthetic model-stage fixtures. Real kernels have been
checked separately as described above; a full fresh target search has not been
run for packaging.
