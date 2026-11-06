# Matched ablations

All recipes call the same scientific modules as the primary task. Bind target,
independent Stage1 run and fitting order explicitly; never select a new target
from the final evaluation results. `swaap run` performs construction, matched
Clean/poison fitting and paired evaluation for one specified latent condition.
Use a new output directory for every condition.

## Prescribed random target, shared support

The recorded random-target seed, exact reference-target tensor SHA and learned
support are bound by the order helper. This condition is different from both
random support and the legacy SWAAP (Random) baseline:

```bash
python scripts/bind_order.py --config configs/tdmpc2/humanoid-walk_random_target_shared_support.yaml --run-index 1 --order hold1 --output outputs/hw-random-target.yaml
swaap run --config outputs/hw-random-target.yaml --artifact-root artifacts --output outputs/hw-random-target
```

Use the corresponding Push/Dog-Run recipe, run indices1–3 and hold1–3 for the
retained27 fitting conditions. Its unchanged construction uses step10. The CPU
reference generator for H-Walk run1 reproduces the closed prescribed tensor SHA
exactly. A user-supplied new seed defines a different experiment.

For random **support** with the learned target, use
`humanoid-walk_swaap_random_support.yaml` and its explicit construction seed. For
legacy SWAAP (Random), bind the ordinary task/run with `bind_order.py`, then set
`method=swaap_legacy_random`. That procedure draws CUDA reference-label noise
before NumPy Bernoulli support; it does not impose exactly500 selected rows.
Random500 remains a separate implementation and has separately source-bound
construction metadata.

## Exact tied trajectory consistency

```bash
python scripts/bind_order.py --config configs/tdmpc2/humanoid-walk_exact_tied_trajectory.yaml --run-index 1 --order hold1 --output outputs/hw-tied.yaml
swaap run --config outputs/hw-tied.yaml --artifact-root artifacts --output outputs/hw-tied
```

`configs/tdmpc2/trajectory_conditions.json` supplies the exact five retained
conditions: H-Walk runs1–3 and Dog-Run runs1–2. The helper verifies the support file
and binds its original ordered250 intermediate states, target, held-out roots,
alpha.9 and fraction.05. Each variable reaches both the predecessor successor
label and the next transition's current-state field through the actual
full-buffer victim MSE. There are500 stored-field writes, terminal/reset
exclusions are preserved, and the development-exposed contexts remain labeled.
JSON supports preserve the exact int64 values/order; the source Torch identity is
also recorded where the original container differs. No support is reselected.

## Fixed500 buffer size

The fixed500 experiment preserves the original poisoned5k prefix and appends
certified Original-policy Clean rows. It does not reconstruct a poison using a
changed ratio. The public preparation routine consumes the exact20k append. The paired
collection manifest records its40 episodes and frozen Original provenance; see
the `buffer-append` profile in the asset guide. Install it alongside `minimal`
for run1, or alongside `main` for all three published runs. After binding those inputs:

```bash
swaap prepare --config configs/tdmpc2/humanoid-walk_fixed500_buffer25000.yaml --set assets.clean=asset:tdmpc2.humanoid-walk.clean-bank --set assets.poison=asset:tdmpc2.humanoid-walk.main-run1.poison --set assets.support=asset:tdmpc2.humanoid-walk.main-run1.support --set assets.extra_clean=asset:tdmpc2.humanoid-walk.certified-append --output outputs/hw25k
```

The recipes default to the first retained run and exact certified append.
The generated `clean.pt` and `poison.pt` supply matched fits; the full literal
fit/evaluation commands are in `provenance/PAPER_TO_CODE.csv`. For10k/15k/20k, use
the corresponding recipe, which takes a certified prefix of the same append.
Keep every original support row fixed; report one deviation scalar per buffer,
then the declared run/order aggregation. Fixed-ratio25k, changed update exposure
and balanced sampling remain different diagnostics, not this table.

## Other retained comparisons

TRIM uses `<task>_trim.yaml`, separately fitting/evaluating defended Clean and
poison. MPC changes only the declared sample/horizon settings of evaluation on
frozen endpoints. The exact online PGD files use epsilon.01/.02 at test time;
those perturbations do not become offline poison labels. Concrete example
commands and dependencies are recorded in `provenance/PAPER_TO_CODE.csv`.
