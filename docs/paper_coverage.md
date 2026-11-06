# Paper reproduction coverage

This guide follows the retained experiments in `main (18).tex` and PDF16, whose
identities are in `provenance/manuscript.json`. The35-row
`provenance/PAPER_TO_CODE.csv` maps paper labels to numerical inputs, concrete
recipes, source pins, execution paths and statistical units.

## Numerical reproduction

```bash
swaap reproduce saved --data-root data/saved --output outputs/paper --figures
```

For the code-only snapshot, first install `saved-results` and use
`--data-root artifacts/results/saved`. The command verifies all75 compact source
inputs, recomputes144 displayed main-table cells, the incoming-CUSUM summary,
alpha/fraction grid and retained ablation tables, and produces supported figures.
It does not run a model. The main table includes the later ManiSkill
Original-reference measurements retained in main18; it never substitutes an
older own-rollout residual.

| Paper item | Public recipe or route | Statistical scope |
|---|---|---|
| Primary nine-task comparison | `configs/tdmpc2/<task>.yaml`; published run/order bindings in `published_orders.json` | Three independent Stage1 runs; three nested final fits per run;100 episodes per endpoint/order |
| Final ManiSkill original SWAAP | `configs/maniskill/{lift-cube,pick-cube,stack-cube}.yaml` | Two/three/three contributing Stage1 runs; rejected Lift selection preserved; no DIRMAG substitution |
| Direct and DMP-data | Ordinary task recipe, `method=direct` / `method=raw_dmp` | Direct Return is target-model return; the starred data deviation describes copied successor labels, not a fitted Direct return |
| SWAAP (Random) | Ordinary recipe with `method=swaap_legacy_random`, bound recorded construction seed | Different from Random500, random support and random target on shared support |
| Alpha/fraction grid | `configs/tdmpc2/sensitivity_grid.yaml` declares cells; run the ordinary task recipe for each |498 run-level conditions,27 reused; no new target selection |
| Incoming CUSUM | [Detector collection/calibration/scoring](detectors.md) | Once per poison; all stored trajectories; independent Clean FPR; no fit-order detector replication |
| TRIM | `configs/tdmpc2/<task>_trim.yaml` | Matched defended Clean utility and poison recovery, reported separately |
| Shared-support random target | `configs/tdmpc2/<task>_random_target_shared_support.yaml` | Same learned support; prescribed random-target seed; step10; three independent runs |
| Top/random support | `humanoid-walk_swaap_random_support.yaml` | Same learned target; changes support only |
| Iterative online PGD | `mw-push_online_pgd_0p01.yaml`, `_0p02.yaml` | Test-time perturbation, distinct from offline .015/.05 data controls |
| MPC sensitivity | `mpc_grid.yaml` supplies settings to the ordinary H-Walk evaluator | Evaluation only, same frozen endpoints and roots |
| Exact trajectory consistency | `<task>_exact_tied_trajectory.yaml` | Three H-Walk and two Dog-Run rows, each with three nested fits; clean initialization, actual full-buffer tied-variable gradient |
| Fixed500 buffer size | `humanoid-walk_fixed500_buffer{10000,15000,20000,25000}.yaml` | Keep original poisoned prefix/support and append certified Clean data; fixed-ratio and sampler diagnostics stay separate |
| Gray-box detailed | `configs/graybox/soccer_artifact.yaml` | Five released MT80 endpoints;100 paired episodes on Soccer/Push; episode variation is not independent attacks |
| Gray-box breadth | `configs/graybox/mt80_breadth.yaml` | Evaluation only on five endpoints;50 paired episodes per task; separate from detailed100 |
| DINO-WM | `configs/dino/pusht_artifact.yaml`, `pusht_faithful.yaml` | Faithful poison/alpha0 distinct; one target/bank, three nested fitting seeds,20 test goals |
| Stage realization negatives | Saved inputs, no automatic new run | Development contexts retain their actual budgets and negative results |
| Hardware/cost | Receipt-derived saved tables | Allocated GPU-hours are not per-stage latency; absent timers remain unavailable |

## Executable scientific paths

The README contains a complete released-target example using `swaap run`.
[Fresh experiments](reproduction.md) describe collection through target selection,
construction, matched fitting, evaluation and aggregation. [Family workflows](families.md)
provide separately executable raw gray-box and DINO paths. Breadth evaluation
has no hidden target optimization, construction or victim fitting. A fresh gray-box
surrogate does not recreate the historical released target's missing producer
state. See [testing](testing.md) for actual real-model checks and their limits.

Use explicit stage output paths for newly generated inputs. A declared grid is
not a scientific worker configuration, and a selected-target example does not
promise that the same iteration will qualify under a new random seed.

## Figures and paper synchronization

The supplied PDF includes the fresh three-task main figure, full-grid figure,
target-development scatters, TRIM and deviation figures. Their numerical inputs
and plotting bodies are provided. For the full grid, the red line is the
**Clean-fitted model's error on its own rollouts**, as the author-adopted main18
caption states. The historical CSV column name `Clean_delta_m_ref` is retained to
preserve input identities; the plot legend correctly says `Clean delta_m`.
No plotted values were replaced by another reference.

The retained historical Stage1 curve PNG and PCA PNGs are **archived images**:
the exact original training-series/PCA input arrays have not been established in
the release. They are not produced by `reproduce saved`, and their filenames do
not establish a reproducible numerical recipe. Source recovery or explicit author
removal remains a paper synchronization decision. No new curves or PCA arrays are
invented to fill that gap.

The results package and this code do not edit the manuscript. The final author
source/PDF, included figure bytes and submission/anonymity mode need a final
comparison before a paper-version tag is made.
