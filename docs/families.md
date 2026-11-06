# Gray-box and DINO-WM experiments

These families use the same command interface as latent TD-MPC2, with separate
encoders, data layouts and scientific kernels. Importing either family starts
no model, simulator or experiment. Paths prefixed `config:` refer to files beside
the selected recipe; generated artifact paths are relative to the caller.

## Installation and assets

Use the family-specific environments in [installation.md](installation.md).
DINO's exact Torch/vision wheel URLs, sizes and SHA256 checksums are recorded in
`environments/dino-wheels.json`; `environments/install-dino.sh` creates a separate
Python3.11.13 environment and installs its pinned dependency closure.

Install the corresponding local asset bundle after obtaining its archive files:

```bash
swaap bundles verify graybox --bundle-dir bundles
swaap bundles install graybox --bundle-dir bundles --root artifacts
swaap bundles verify dino --bundle-dir bundles
swaap bundles install dino --bundle-dir bundles --root artifacts
```

The bundle manifest gives exact archive names, sizes, checksums and component
relationships. The separate `dino-endpoints` bundle contains optional fitted
models. Constructing or fitting a new poison does not require all saved endpoints.
No asset is silently replaced with a newer upstream checkpoint.

## Legacy MetaWorld and gray-box installation

Create the exact Python3.9.21/CUDA12.4 TD-MPC2 base using the commands in
[the latent-family guide](latent.md). In that separate environment, install the
legacy simulator layer:

```bash
bash environments/install-graybox.sh "$PWD/.runtime-graybox"
source .runtime-graybox/activate-graybox.sh
```

The installer uses MetaWorld commit `04be337a12305e393c0caf0cbf5ec7755c7c8feb`,
MuJoCo210, `mujoco-py==2.1.2.14`, `Cython==0.29.37`, and `gym==0.23.1`.
It downloads the official MuJoCo archive and verifies its complete SHA-256;
`graybox-native-assets.json` also fixes the Ubuntu22.04 x86_64 OSMesa/GL
build-package bytes. These packages are extracted into the chosen user prefix,
without changing system libraries. A C compiler, `dpkg-deb`, and the Ubuntu
runtime libraries needed by OSMesa (including LLVM15 and `libglapi`) must be
available; on a minimal Ubuntu installation use
`sudo apt install build-essential libllvm15 libglapi-mesa libdrm2` first.
The interface is state-only: the package imports its CPU extension but the
experiment does not use rendering or camera observations. Current MetaWorld or
MuJoCo releases are different experiment environments and are not substituted.
The same installation supports the ordinary nine-task MetaWorld recipes.

## Gray-box: reproduce the released endpoints

The gray-box experiment constructs successor observations with a single-task
Soccer surrogate and fits the separately encoded MT80 victim. The artifact
route evaluates five frozen endpoints: Original and matched Clean/SWAAP fits
at learning rates2.5e-5 and5e-5.

```bash
swaap evaluate --config configs/graybox/soccer_artifact.yaml --output outputs/graybox-detailed
swaap aggregate --config configs/graybox/soccer_artifact.yaml --set assets.evaluation=outputs/graybox-detailed/frozen_evaluation.json --output outputs/graybox-summary
```

This evaluates Soccer and Push on their exact100-root panels. Return variation
is episode variation from one source target/poison and one victim fit per method
and learning rate; it is not variation over independent attacks.

The80-task breadth evaluation loads those same five endpoints and performs no
construction, target generation or fitting:

```bash
swaap evaluate --config configs/graybox/mt80_breadth.yaml --output outputs/graybox-breadth
swaap aggregate --config configs/graybox/mt80_breadth.yaml --set assets.evaluation=outputs/graybox-breadth/breadth.json --output outputs/graybox-breadth-summary
```

For one task, add `--set 'evaluation.tasks=[mw-soccer]'`. Breadth uses50 paired
episodes per endpoint/task; it must not be pooled with the detailed100-root panel.
All signed task reductions are retained, including improvements.

## Gray-box: reconstruct or refit

The raw-space constructor differentiates through the frozen single-task encoder.
It matches the full5000-row mean-MSE gradient to a50000-row latent reference.
Only the native top500 target-residual successor observations can change. The
published configuration uses alpha0,5000 updates, raw step1 and coordinate-relative
epsilon .0045. Zero anchor coordinates stay zero; there is no radius floor or
raw[0,1] clamp.

```bash
swaap prepare --config configs/graybox/soccer_reconstruct.yaml --output outputs/graybox-inputs
swaap construct --config configs/graybox/soccer_reconstruct.yaml --output outputs/graybox-poison
swaap fit --config configs/graybox/soccer_refit.yaml --set method=clean --output outputs/graybox-clean
swaap fit --config configs/graybox/soccer_refit.yaml --set assets.poison=outputs/graybox-poison/poison.pt --output outputs/graybox-swaap
swaap evaluate --config configs/graybox/soccer_refit.yaml --set method=clean --set assets.checkpoint=outputs/graybox-clean/fit.pt --output outputs/graybox-clean-eval
swaap evaluate --config configs/graybox/soccer_refit.yaml --set assets.checkpoint=outputs/graybox-swaap/fit.pt --output outputs/graybox-swaap-eval
```

`prepare` verifies the released raw/reference banks. Their original producer
random state is not retained, so exact input reproduction uses the supplied
banks. MT80 independently
encodes the raw data with its Soccer embedding68, padding4 actions to6. Fitting
is fresh dynamics-only Adam, E10/B256/U200, sequential order each epoch. Actor,
encoder, reward and value weights remain fixed. A new fit at5e-5 uses an explicit
`--set fitting.learning_rate=5e-5`; retain its separately matched Clean fit.

`configs/graybox/soccer_fresh_surrogate.yaml` exposes the shared executed Stage1
for a new single-task target. Its `stage1` and `select` commands use fresh
eligibility/Pareto selection and report rejections. They do not reproduce the
historical target identity. A newly selected target can be supplied to the raw
constructor with `--set assets.target=... --set assets.support=null`; null support
requests a new target's native top500 rather than reusing the published mask.
Target metadata must bind the same Original checkpoint. This is a new experiment
over the explicitly supplied banks, separate from released-artifact reproduction.

## DINO-WM: released poison and matched fitting

```bash
swaap prepare --config configs/dino/pusht_artifact.yaml --output outputs/dino-inputs
swaap fit --config configs/dino/pusht_artifact.yaml --set method=clean --output outputs/dino-clean99
swaap fit --config configs/dino/pusht_artifact.yaml --output outputs/dino-swaap99
swaap evaluate --config configs/dino/pusht_artifact.yaml --set method=clean --set assets.checkpoint=outputs/dino-clean99/fit.pt --output outputs/dino-clean99-eval
swaap evaluate --config configs/dino/pusht_artifact.yaml --set assets.checkpoint=outputs/dino-swaap99/fit.pt --output outputs/dino-swaap99-eval
```

Repeat seeds100/101 in separate directories with the same poison and20 goals.
These are three nested fits of one target/bank/poison, not three independent
attacks. Alpha0 is a separate arm: set `assets.poison=asset:dino.pusht.poison-alpha0`.
For Original evaluation use `--set method=original`; fitted methods always require
an explicit checkpoint and cannot fall back to Original.

To reconstruct the ordinary SWAAP poison rather than import it:

```bash
swaap construct --config configs/dino/pusht_faithful.yaml --output outputs/dino-poison
swaap fit --config configs/dino/pusht_faithful.yaml --set assets.poison=outputs/dino-poison/poison.pt --output outputs/dino-reconstructed-fit
```

Each row has three histories,196 tokens and404 features. The32 editable windows
are restricted to the first80 of320; only successor features0:394 are editable.
Action features394:404 and every unselected row stay fixed. All404 features enter
the full-buffer mean-MSE gradient. The stealth regularizer uses per-window RMS
over196x394, not an unnormalized L2 norm.

Construction uses full-model BF16, predictor-only gradients, dropout off, chunks4
with FP32 weighted gradient accumulation, FP32 editable masters, clean
initialization and K5000/step10/epsilon1. Projection preserves the same clean box
for FP32 and BF16-consumed labels. Fitting is BF16 E10/B16/U200, predictor-only
fresh AdamW2e-5 with weight decay .01 and the native shuffled loader. A saved
BF16 endpoint is loaded into an FP32 Original deployment template. Direct
replaces only the FP32 predictor and preserves Original's module modes/flags.

For a fresh target and observed bank, run these stages explicitly:

```bash
swaap stage1 --config configs/dino/pusht_fresh.yaml --output outputs/dino-new-targets
swaap select --config configs/dino/pusht_fresh.yaml --set assets.target=outputs/dino-new-targets/target_25.pt --output outputs/dino-new-selection
swaap prepare --config configs/dino/pusht_fresh.yaml --output outputs/dino-new-data
swaap construct --config configs/dino/pusht_faithful.yaml --set assets.target=outputs/dino-new-targets/target_25.pt --set assets.clean=outputs/dino-new-data/clean.pt --output outputs/dino-new-poison
swaap fit --config configs/dino/pusht_faithful.yaml --set assets.clean=outputs/dino-new-data/clean.pt --set assets.poison=outputs/dino-new-poison/poison.pt --output outputs/dino-new-fit
```

Fit a separately matched Clean with the same new bank and seed, and evaluate
both saved endpoints as above. For incoming metrics, bind both
`assets.clean=outputs/dino-new-data/clean.pt` and
`assets.reference_predictions=outputs/dino-new-data/reference_predictions.pt`.
These predictions use the same frozen FP32 Original; the saved window origins
join every fitting trajectory. The new Stage1 output is passed explicitly;
the released target is not substituted for it.

The separate `pusht_fresh.yaml` exposes the source Stage1 update and fixed-length
observed collection. Stage1 saves updates1/10/25/50; the retained target is fixed
at25. Collection retains ten genuine grouped MPC actions and a genuine unexecuted
boundary action, producing eight history3 windows per trajectory. Controlled,
reference and untouched windows remain80/80/240. The optional old MP4-dataset
loader is not needed by these recorded-goal/observed-collection paths; its video
reader must be installed separately if that different input route is used.

## Incoming-buffer CUSUM

The detector always uses pretrained Original residuals before victim fitting.
Gray-box collection follows the single-task Soccer Original policy and is scored
by the separate MT80 Original under embedding68. DINO uses FP32 Original,
196x394 successor-state fields and eight ordered windows per trajectory.

Use the released valid cohorts without recollecting them:

```bash
swaap prepare --config configs/graybox/soccer_detector_saved.yaml --output outputs/graybox-cohorts
swaap calibrate-detector --statistics outputs/graybox-cohorts/statistics.npz --threshold outputs/graybox-cohorts/threshold.npz --family graybox --output outputs/graybox-cohorts/calibration.json
swaap prepare --config configs/dino/pusht_detector_saved.yaml --output outputs/dino-cohorts
swaap calibrate-detector --statistics outputs/dino-cohorts/statistics.npz --threshold outputs/dino-cohorts/threshold.npz --family dino-wm --output outputs/dino-cohorts/calibration.json
swaap detect-buffer --config configs/dino/pusht_artifact.yaml --set assets.calibration=outputs/dino-cohorts/calibration.json --set assets.clean_test=outputs/dino-cohorts/clean_test.npz --output outputs/dino-alpha09-detection
swaap detect-buffer --config configs/dino/pusht_artifact.yaml --set alpha=0 --set assets.poison=asset:dino.pusht.poison-alpha0 --set assets.calibration=outputs/dino-cohorts/calibration.json --set assets.clean_test=outputs/dino-cohorts/clean_test.npz --output outputs/dino-alpha0-detection
swaap detect-buffer --config configs/graybox/soccer_refit.yaml --set assets.calibration=outputs/graybox-cohorts/calibration.json --set assets.clean_test=outputs/graybox-cohorts/clean_test.npz --output outputs/graybox-detection
```

The derived scorer-schema NPZ files preserve the released residual values
and bind the supplied trajectory IDs or exact environment/planner seed pairs.
Gray-box uses300 statistics/500 threshold/1000 independent clean trajectories;
DINO uses200/200/400. Both frozen DINO poisons use the same calibration. All50
raw gray-box trajectories or40 DINO trajectories enter scoring, including those
with no modified row. Nominal thresholds and realized independent Clean FPR
remain separate. Construction's BF16 deviations are labeled diagnostics; DINO's
paper delta_d uses canonical FP32 labels and FP32 Original predictions.

For new collection, use `collect-detector` with the corresponding
`*_detector_collect.yaml`, `detector.role=statistics`, then `threshold`, freeze
with `calibrate-detector`, and collect `clean_test` with `assets.calibration`
pointing to that frozen file. The supplied manifests fix disjoint roots/records.
A collection smoke adds `--set collection.maximum_trajectories=1`; it writes the
same residual schema but is not a statistically meaningful calibration and
cannot pass the full-cohort size gates.

`configs/{graybox,dino}/smoke.yaml` are integration-prefix recipes: two
constructor updates, short matched fits and one complete episode/goal per
endpoint. Their outputs are engineering tests, not published efficacy results.
