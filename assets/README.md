# Model, data and numerical-result bundles

Asset bundles are separate from the source distribution. Each named profile lists
its complete archive dependencies in `bundle_manifest.json`. Identical content is
stored once across the archive components; profiles share components where
appropriate. A single-task example does not require every fitted checkpoint.

| Profile | Required archive bytes | Components | Asset aliases | Contents |
|---|---:|---:|---:|---|
| `saved-results` | 33,095,680 | 1 | 76 | Complete portable CPU reproduction inputs: genuine retained numerical arrays/tables plus their own checksum manifest; no pretrained weights or raw states. |
| `minimal` | 306,206,720 | 2 | 9 | One fixed-target Humanoid-Walk SWAAP construction, matched fitting and endpoint evaluation; one exact order, without unrelated fitted models. |
| `main` | 3,984,824,320 | 6 | 141 | Nine-task and final ManiSkill published Original/bank/reference, selected target, support and poison inputs; fit endpoints from these inputs or obtain the separate endpoint profile. |
| `graybox` | 294,635,520 | 2 | 14 | Frozen Soccer surrogate, raw bank/reference/support/poison, all five MT80 evaluation endpoints and released incoming-detector cohorts. |
| `dino` | 2,838,456,320 | 4 | 18 | Faithful DINO-WM construction/fitting/evaluation inputs: Original, fixed target, history bank/reference, HIQL, sampler/goals, frozen alpha0/alpha.9 labels and detector cohorts. No fitted endpoints are required. |
| `endpoints` | 1,593,559,040 | 3 | 243 | Optional nine-task full matched Clean/SWAAP fitted endpoints and exact saved permutations; unnecessary for the minimal example. |
| `dino-endpoints` | 4,534,906,880 | 6 | 27 | Optional nine faithful DINO-WM fitted endpoints together with their complete input/deployment dependencies; fit them yourself instead of downloading this profile. |
| `buffer-append` | 83,814,400 | 1 | 2 | Exact certified 20,000-transition Humanoid-Walk append and original collection manifest. Install `minimal` separately for a single-run comparison or `main` for all three runs. |
| `historical-figures` | 1,679,360 | 1 | 3 | Exact historical Stage-1/PCA PNGs included by the manuscript. Numerical source arrays were not recovered, so these images cannot be regenerated from the release. |

The complete distribution contains 19 archive components totaling
10,289,039,360 bytes and 489 unique content objects. Do not add the profile sizes
to estimate storage: several profiles refer to the same components. Individual
components are below 1 GB. `asset_manifest.json` records each scientific file's
exact byte count, SHA-256, model version and upstream acquisition where available.

## Install a supplied distribution directory

Place the profile's component archives in one directory, for example `bundles/`,
and use the manifest shipped with the same source version:

```bash
swaap bundles list --bundle-dir bundles
swaap bundles verify minimal --bundle-dir bundles
swaap bundles install minimal --bundle-dir bundles --root artifacts
```

Verification checks the archive and every content object before exposing the
asset aliases. Installation preserves the original model/data bytes and retains
the supplied upstream notices. Existing mismatched files or partial imports are
preserved and reported; they are not silently overwritten.

For the standalone CPU numerical path, including when the source snapshot omits
the larger numerical files:

```bash
swaap bundles install saved-results --bundle-dir bundles --root artifacts
swaap reproduce saved --data-root artifacts/results/saved --output outputs/paper --figures
```

For real scientific execution, install `minimal` and follow the repository's
fixed-target Humanoid-Walk example. It includes Original, the 5,000-row Clean and
50,000-row reference banks, the exact selected target/support/poison, one matched
Clean/SWAAP fitted endpoint pair and its fitting order. `main` supplies all nine
TD-MPC2 and final ManiSkill construction inputs. `endpoints` is optional: a reader
can fit endpoints from the supplied inputs instead. Main-table SWAAP (Random)
uses its separate recorded seed inventory and unchanged construction procedure;
it is not an alias for Random500.

`dino` contains the exact HIQL policy, fixed Stage-1 sampler and final goal
records, history-dependent banks, Original/target models, frozen poisons and valid
detector cohorts. It includes **no fitted endpoints**. Use `dino-endpoints` only
when evaluating the nine supplied fitted models instead of fitting them. The
DINO-WM and raw-observation gray-box paths use separate implementations and
reference encoders; do not exchange their inputs with latent TD-MPC2 buffers.


For the larger-buffer example, install `minimal` and `buffer-append`; the append
profile itself contains only the certified additional rows and their original
collection manifest. Use `main` instead of `minimal` when reproducing all three
Humanoid-Walk runs. These are the original collection bytes; historical provenance
strings in that manifest are not executable path dependencies.

## Upstream pretrained models and individual imports

Official TD-MPC2 Originals are also available from their checksum-pinned
[upstream model repository](https://huggingface.co/nicklashansen/tdmpc2), and the
DINOv2 backbone has its recorded official URL. For these individual public inputs:

```bash
swaap assets download tdmpc2.humanoid-walk.original --root artifacts
swaap assets verify tdmpc2.humanoid-walk.original --root artifacts
```

`assets import ID --file PATH` accepts an existing exact artifact and verifies its
identity. It does not substitute a similarly named newer model. Bundle network
downloads use only HTTPS endpoints actually present in the bundle manifest; a
local archive directory remains sufficient for verification and installation.

## Rebuilding distribution archives

The packaging script uses explicit file mappings and never searches a machine or
loads a model:

```bash
python scripts/pack_assets.py build --manifest assets/asset_manifest.json \
  --profiles assets/bundle_profiles.json --sources /your/explicit-source-map.json \
  --notices assets/license_notices.json --output /your/new-distribution
```

The source map is `{"sources":{"asset-id":"/your/file"}}`, or it may use exact
content hashes as keys. Keep machine-specific source maps outside the repository.
The command verifies source bytes while creating deterministic content-addressed
archives, then writes archive hashes and profile dependencies. It will not
replace an existing output directory. Fresh collection or training creates a new
experiment; it does not recover the published artifact bytes automatically.

See `docs/licensing.md` and `THIRD_PARTY_NOTICES.md` for software, model and
data distinctions. This compact public distribution is not a backup of every
raw trajectory or developmental experiment.
