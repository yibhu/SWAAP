# Incoming-buffer residual CUSUM

This detector scores incoming data **before fine-tuning**. It predicts successors
with frozen Original and uses raw residual L2 sequences. Neither fitted-model
rollout error nor a development deviation coordinate is the detector input.
Statistics, threshold calibration, independent Clean testing and incoming poison
must be disjoint. Every stored trajectory enters the denominator, including
trajectories with no modified row.

| Family | Statistics trajectories | Threshold trajectories | Independent Clean test | Incoming trajectory unit |
|---|---:|---:|---:|---|
| H-Walk and MW-Push | 500 | 500 | 1000 | 500/100 transitions respectively |
| Other seven primary tasks | 300 | 500 | 1000 | 500 DMC or100 Myo/MetaWorld transitions |
| ManiSkill | 500 | 500 | 1000 | 100 outer decisions, action repeat2 |
| Raw gray-box | 300 | 500 | 1000 | Soccer source-policy trajectory, MT80 reference |
| DINO-WM | 200 | 200 | 400 | eight ordered history3 windows |

The recurrence uses residual median/MAD, drift factor6, split-conformal threshold
rank `ceil((n+1)*(1-FPR))` and strict `score > threshold`. A rank beyond the
calibration sample gives an infinite threshold. Tiny cohorts cannot be relabeled
as a paper calibration. Report nominal thresholds separately from realized
independent Clean FPR. The scorer also returns AUC, McClish-standardized pAUC at
.05/.10, TPR, Wilson intervals and bootstrap intervals.

## A complete new latent-family collection

This example declares a **new** H-Walk experiment. It does not claim to recover
the historical root streams. `make_detector_roots.py` derives explicit roots
without a model call, binds the verified Original SHA, and writes mutually
disjoint Original-policy and new-data collection panels:

```bash
python scripts/make_detector_roots.py --task humanoid-walk --seed 1701 --original tdmpc2.humanoid-walk.original --artifact-root artifacts --output outputs/hw-roots
swaap collect-detector --config configs/tdmpc2/humanoid-walk.yaml --artifact-root artifacts --set assets.roots=outputs/hw-roots/statistics.json --set detector.role=statistics --output outputs/hw-statistics
swaap collect-detector --config configs/tdmpc2/humanoid-walk.yaml --artifact-root artifacts --set assets.roots=outputs/hw-roots/threshold.json --set detector.role=threshold --output outputs/hw-threshold
swaap calibrate-detector --statistics outputs/hw-statistics/residuals.npz --threshold outputs/hw-threshold/residuals.npz --family tdmpc2 --task humanoid-walk --output outputs/hw-calibration.json
swaap collect-detector --config configs/tdmpc2/humanoid-walk.yaml --artifact-root artifacts --set assets.roots=outputs/hw-roots/clean_test.json --set detector.role=clean_test --set assets.calibration=outputs/hw-calibration.json --output outputs/hw-clean-test
swaap prepare --config configs/tdmpc2/humanoid-walk.yaml --artifact-root artifacts --set data.mode=explicit_roots --set assets.collection_roots=outputs/hw-roots/collection.json --output outputs/hw-new-data
```

The optional explicit-root preparation collects one seeded trajectory per call.
Its RNG batching differs from the historical batched preparation; it is a new
stochastic data collection with unchanged trajectory math. For a newly constructed
poison of `hw-new-data/clean.pt`, use that bank's emitted provenance:

```bash
swaap detect-buffer --config configs/examples/humanoid-walk.yaml --artifact-root artifacts --set assets.clean=outputs/hw-new-data/clean.pt --set assets.poison=outputs/hw-new-poison/poison.pt --set assets.trajectory_roots=outputs/hw-new-data/clean_trajectory_roots.json --set assets.calibration=outputs/hw-calibration.json --set assets.clean_test=outputs/hw-clean-test/residuals.npz --output outputs/hw-incoming-metrics
```

`hw-new-poison/poison.pt` is the explicit output of `swaap construct` using this
new bank and its reference, following the [fresh experiment](reproduction.md).
It must not be a poison from another bank. The main model bundle does not include
the nine-task detector's raw historical calibration cohorts. Exact published
aggregate detector results remain available in the saved numerical bundle;
new collection is a distinct reproduction path.

For ManiSkill, the supplied `*_detector_{stats,threshold,clean_test}_roots.json`
files provide the recorded disjoint calibration panels. Use the matching
ManiSkill task recipe and `--family maniskill`. Do not reuse final fitting roots
as calibration or relabel trajectories to evade an overlap check.

## Gray-box and DINO-WM

[Family workflows](families.md#incoming-buffer-cusum) give complete commands for
released cohorts and both frozen DINO poisons. The released NPZ arrays preserve
the actual residual values; preparation writes the common scorer schema.
Gray-box collection uses the single-task Soccer Original policy, then the
**different MT80 Original encoder and task68** for residuals. DINO uses FP32
Original prediction, eight fixed windows and196x394 editable successor features.
Construction's BF16 deviations are separate diagnostics.

For fresh collection use `configs/graybox/soccer_detector_collect.yaml` or
`configs/dino/pusht_detector_collect.yaml`, selecting statistics, threshold and
clean_test in that order. Their manifests define the full recorded roots/goals;
`collection.maximum_trajectories=1` produces an engineering sample of the same
schema and cannot qualify a full detector calibration. No command automatically
recollects800 trajectories when using released DINO cohorts.
