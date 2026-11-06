# Code, checkpoint and data terms

Software, pretrained models and collected scientific data have distinct rights.
A source-code license does not automatically license every checkpoint, mesh or
trajectory produced with that software. File identities and acquisition routes
are recorded in `assets/asset_manifest.json`; bundle contents are recorded in
`assets/bundle_manifest.json`.

| Material | Applicable upstream terms and notices |
|---|---|
| Adapted TD-MPC2 code | MIT; retain `licenses/TD-MPC2-MIT.txt`. Its DrQ-v2 and DreamerV3 adaptations retain their identified notices. |
| Official TD-MPC2 pretrained models | MIT as declared by the official model repository. The pinned download revision and individual SHA-256/byte counts are recorded. Retain this upstream notice in derived checkpoint distributions. |
| DINO-WM source | Its own source is MIT. Incorporated code retains file-specific terms, including Apache-2.0 Sonnet material, MIT vision-transformer/R3M helpers and the I-JEPA adaptation described below. |
| DINOv2 backbone | Apache-2.0; retain the upstream notice and exact backbone identity. |
| HIQL framework adaptations | OGBench MIT source notices apply to the adapted implementation. They do not establish a separate license for the custom trained HIQL checkpoint. |
| DMControl, MyoSuite and MetaWorld | Respect their separately distributed package and asset terms. The retained package profiles do not replace upstream licenses. |
| ManiSkill2 0.5.3 / SAPIEN 2.2.2 runtime | [Versioned ManiSkill2 code license](https://raw.githubusercontent.com/mani-skill/ManiSkill/v0.5.3/LICENSE) is Apache-2.0; the [same version's asset statement](https://raw.githubusercontent.com/mani-skill/ManiSkill/v0.5.3/README.md#license) identifies CC BY-NC 4.0 for its assets. Simulator meshes and complete environments are not included in the model/data bundles. |
| SWAAP-authored code, custom checkpoints and collected buffers | These retain their respective author-supplied terms; do not infer a grant from a different upstream component. The source snapshot makes no blanket license assignment to these different categories. |

## I-JEPA-derived portions

`src/swaap/dino/_vendor/models/proprio.py` identifies an adaptation from I-JEPA's
`src/models/vision_transformer.py`. The retained positional-embedding helpers
`get_1d_sincos_pos_embed` and `get_1d_sincos_pos_embed_from_grid` follow that
implementation. The upstream I-JEPA license is **CC BY-NC 4.0**. Its text is
retained in `licenses/I-JEPA-CC-BY-NC-4.0.txt`; do not relabel these adapted portions
as uniformly MIT. This statement does not automatically assign the same license
to all DINO checkpoints, the HIQL policy or collected data.

The source origins and copyright notices are collected in
`THIRD_PARTY_NOTICES.md`. The bundle manifest carries these notices, and the
bundle installer retains a copy under the artifact root's `licenses/` directory.

## Scientific artifacts

The bundles contain checksum-bound models, optimized labels, selected indices,
reference inputs, calibration arrays and compact numerical results. They exclude
private execution logs, complete simulator installations, Python environments and
unnecessary historical archives. The exact published bytes remain distinct from
newly collected buffers or newly trained models. Different authors' permissions
and terms must not be replaced by a single inferred repository-wide grant.

Use the profile and asset manifests to identify what a file is and how it is
acquired. The normal entry point is the small named bundle described in
`assets/README.md`; each artifact retains its file-specific rights metadata.
