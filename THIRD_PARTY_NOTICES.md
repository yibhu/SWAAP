# Third-party notices

SWAAP reuses/adapts TD-MPC2 scientific/model code by Nicklas Hansen and
contributors and DINO-WM code by Gaoyue Zhou and contributors. Their MIT notices
are preserved in `licenses/TD-MPC2-MIT.txt` and `licenses/DINO-WM-MIT.txt`.
Those licenses apply to their respective material and remain separate from the
terms assigned to SWAAP-authored code, custom models and collected data.

The installation also uses separately distributed packages including PyTorch,
NumPy, SciPy, scikit-learn, Matplotlib, Hydra/OmegaConf, DMControl/MuJoCo,
Meta-World, MyoSuite, ManiSkill2/SAPIEN and DINOv2. Their own license and attribution
requirements remain applicable. See `docs/licensing.md` for verified primary
license references and the ManiSkill asset/code distinction. Dependency versions
are specified separately by runtime family.

The gray-box path uses TD-MPC2 surrogate and multi-task victim models with distinct
encoders. The DINO-WM path uses history-dependent observation/action features and a
custom trained HIQL auxiliary policy. Neither of these custom experimental
artifacts acquires a redistribution license merely by using MIT/Apache code.

No third-party simulator meshes, pretrained model weights, raw trajectories,
review material or private experimental logs are bundled with these notices.

## Notices in adapted scientific dependencies

The following file-level origins are identified by the retained source comments.
These restrictions are not replaced by DINO-WM's top-level MIT license:

| Retained material | Upstream origin and terms | Notice |
|---|---|---|
| TD-MPC2 math helpers | [DreamerV3](https://github.com/danijar/dreamerv3), MIT permission text | `licenses/DreamerV3-MIT.txt` |
| TD-MPC2 layers and DMControl wrapper adaptations | [DrQ-v2](https://github.com/facebookresearch/drqv2), MIT | `licenses/DrQv2-MIT.txt` |
| DINO-WM `models/proprio.py` positional-embedding helper adaptation | [I-JEPA](https://github.com/facebookresearch/ijepa), **CC BY-NC 4.0** | `licenses/I-JEPA-CC-BY-NC-4.0.txt`; retain the noncommercial terms for the adapted helpers; do not assign that license automatically to every checkpoint or dataset. |
| DINO-WM vision transformer adaptation | [vit-pytorch](https://github.com/lucidrains/vit-pytorch), MIT | `licenses/ViT-PyTorch-MIT.txt` |
| DINO-WM vector quantizer | [Sonnet](https://github.com/google-deepmind/sonnet), Apache-2.0; original Sonnet copyright header retained | `licenses/Sonnet-Apache-2.0.txt` |
| DINO-WM vendored R3M encoder helpers | [R3M](https://github.com/facebookresearch/r3m), MIT; source copyright headers retained | `licenses/R3M-MIT.txt` |
| Trained HIQL runtime framework adaptation | [OGBench](https://github.com/seohongpark/ogbench), MIT | `licenses/OGBench-MIT.txt`; this code license does not establish a grant for the custom policy checkpoint. |

DINO-WM's Push-T source also identifies an adaptation from Implicit Behavior
Cloning. Its exact source/version attribution remains an author release-review
item. Do not describe this mixed-origin vendor directory as uniformly MIT or
silently rewrite source code to avoid attribution.

The adopted state-only ManiSkill runtime uses ManiSkill2 0.5.3 with SAPIEN
2.2.2. `licenses/ManiSkill2-0.5.3-Apache-2.0.txt` retains its versioned code
license. The same version identifies its separately obtained simulator assets
as CC BY-NC 4.0; those meshes and complete simulator installations are not
included in the scientific input bundles.
