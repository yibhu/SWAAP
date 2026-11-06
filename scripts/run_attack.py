#!/usr/bin/env python3
"""One fixed-target TD-MPC2 construction, matched fitting and evaluation.

This serial convenience script uses the same functions as the individual CLI
commands. It performs no target search, automatic retries or result selection.
Use a new output directory; partial runs are preserved and never replayed.
"""
import argparse
import json
from pathlib import Path
from swaap.artifacts import AssetStore
from swaap.config import load_config,resolve_assets


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--target',help='Selected target file or asset:ID; defaults to assets.target in the recipe.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,default=Path('artifacts'))
    p.add_argument('--manifest',type=Path)
    p.add_argument('--set',action='append',default=[])
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv);cfg=load_config(a.config,a.set)
    if cfg['family'] not in ('tdmpc2','maniskill'):
        raise ValueError('This convenience script is for the shared latent TD-MPC2 executor. See family instructions for gray-box/DINO.')
    if a.target:cfg.setdefault('assets',{})['target']=a.target
    cfg,assets=resolve_assets(cfg,AssetStore(a.artifact_root,a.manifest))
    root=a.output.resolve()
    if root.exists():raise ValueError('Output exists; no implicit resume or overwrite.')
    if a.dry_run:
        print(json.dumps(dict(status='CONFIGURATION_ONLY_NOT_EXECUTED',stages=[
              'construct','Clean fit','poison fit','Clean evaluate','poison evaluate'],config=cfg,assets=assets),indent=2))
        return 0
    from swaap.workflow import fixed_target
    cfg['output_dir']=str(root)
    fixed_target(cfg)
    print(json.dumps(dict(status='COMPLETE',output=str(root))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
