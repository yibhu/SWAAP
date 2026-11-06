# SWAAP

Reproduction code for **Stealthy World Model Manipulation via Data Poisoning**,
by **Yibin Hu and Zizhan Zheng**.

SWAAP identifies a harmful target dynamics model, then modifies a bounded set of
successor labels so that victim fine-tuning moves toward that target. This
repository includes latent TD-MPC2 experiments, raw-observation transfer to MT80,
and history-dependent DINO-WM experiments on simulation benchmarks.

## Installation

Use Python 3.12 for saved numerical results and CPU plotting:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[saved,plots]'
```

Model execution uses separate environments for incompatible simulator and
backbone versions. Follow [installation](docs/installation.md), then install this
package inside the chosen environment with `python -m pip install --no-deps .`.
Do not install the CPU plotting extras into an older simulator environment.

## Reproduce saved tables and figures

From a source checkout that includes `data/saved`:

```bash
swaap reproduce saved --data-root data/saved --output outputs/paper --figures
```

The command verifies input checksums, recomputes the adopted tables and available
figures, and writes `outputs/paper/tables`, `outputs/paper/figures`, and
`reproduction.json`. It loads no world model and runs no simulator.

The code-only source snapshot excludes numerical assets. Install their separate
bundle when using that snapshot:

```bash
swaap bundles install saved-results --bundle-dir bundles --root artifacts
swaap reproduce saved --data-root artifacts/results/saved --output outputs/paper --figures
```

Historical training-curve/PCA files are distinguished from figures with released
plotting arrays in the [paper reproduction guide](docs/paper_coverage.md).

## Run a real SWAAP example

After installing the TD-MPC2 environment, place the two minimal-bundle archives
listed in `assets/bundle_manifest.json` in `bundles/`, then run:

```bash
swaap bundles verify minimal --bundle-dir bundles --root artifacts
swaap bundles install minimal --bundle-dir bundles --root artifacts
swaap run --config configs/examples/humanoid-walk_smoke.yaml --artifact-root artifacts --output outputs/hwalk-example
```

This example reuses a released H-Walk target and fixed data. It performs two
actual SWAAP construction updates, matched Clean/poison fine-tuning from Original,
saves and reloads both checkpoints, and evaluates one full 500-step episode per
endpoint on the same declared root. It is a reduced engineering example, not an
estimate of paper efficacy. `result.json` contains the paired returns, harm,
identities, and paths to the construction, fitting and evaluation outputs.

For the full fixed-target workload, use `configs/examples/humanoid-walk.yaml`:
5,000 construction updates, ten fitting epochs, and 100 evaluation episodes per
endpoint. A new target search is a separate [fresh experiment](docs/reproduction.md).
Use a new output directory; commands never silently overwrite or resume results.

For one complete fresh run on a primary task, the ordinary serial script
`python scripts/run_fresh_task.py --config configs/tdmpc2/humanoid-walk.yaml --seed 17 --output outputs/hwalk-fresh17`
performs collection, Stage1 admission, target screening/selection and matched final
evaluation. This is the full scientific workload; its stage handoffs have CPU
fixture tests, while the real-model checks use the reduced fixed-target example.
It launches no scheduler or automatic second attempt.

## Three reproduction paths

| Path | Computation | Guide |
|---|---|---|
| Saved results | CPU aggregation and plotting of published numerical inputs | [Paper items](docs/paper_coverage.md) |
| Released artifacts | Verified models/data/poisons, followed by explicitly requested fitting or evaluation | [Assets](assets/README.md), [family workflows](docs/families.md) |
| Fresh experiments | New collection and Stage1, target selection, construction, matched fitting/evaluation | [Experiment guide](docs/reproduction.md), [TD-MPC2/ManiSkill](docs/latent.md) |

Gray-box paper reproduction starts from the released Soccer target. Its 80-task
breadth workflow evaluates the five fixed MT80 endpoints; it performs no hidden
fitting or target search. A newly optimized surrogate is a new experiment.
DINO-WM keeps its own encoders, history windows, fixed action features, precision
and support restrictions. These paths are not interchangeable latent attacks.

## Assets and outputs

```bash
swaap bundles list
swaap assets list --filter tdmpc2.humanoid-walk
swaap assets verify tdmpc2.humanoid-walk.original --root artifacts
```

[Bundle acquisition and verification](assets/README.md) lists exact archives,
sizes, SHA-256 identities and regeneration routes. Content shared by profiles is
stored once. The minimal example does not require all evaluation checkpoints.
Large model/data files stay in `artifacts/`; run outputs stay in `outputs/`.
Both locations are configurable. There are no required scheduler or notification
services.

The main-table random baseline is **SWAAP (Random)**. Random500, random support,
and random target on shared support have separate implementations and recipes.
Incoming-buffer CUSUM is measured before fitting with disjoint clean roles;
rollout prediction error is a different metric. See [detectors](docs/detectors.md).

## Statistics and verification

Average evaluation episodes within a fitting run, then the three fitting runs
within each Stage1 run. Compute the reported mean and sample SD across independent
Stage1 runs. The final ManiSkill counts are two/three/three contributing runs;
the failed third Lift selection remains recorded. Gray-box episode variation and
DINO fitting-order variation are not independent attack replications. Missing
measurements are never zeros, and negative outcomes are retained.

[Testing notes](docs/testing.md) describe the exact environments, short real-model
checks, CPU parity tests and their limits. Fresh RL returns need not match across
all devices. Saved-result replay, short integration tests and complete stochastic
experiments provide different evidence.

## Citation and licensing

Citation metadata is in [CITATION.cff](CITATION.cff). Third-party terms are retained
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`; those grants do
not override other components. See [licensing](docs/licensing.md) for source,
checkpoint and data terms. The supported paper snapshot is identified in
`provenance/manuscript.json`.
