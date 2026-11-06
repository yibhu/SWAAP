"""Ordinary, explicit research commands. Importing this module runs nothing."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import yaml
from . import __version__


def parser():
    p = argparse.ArgumentParser(prog='swaap', description='SWAAP scientific reproduction. No command silently resumes or overwrites outputs.')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='command', required=True)
    assets = sub.add_parser('assets', help='List, verify, import, or download exact named assets.')
    assets.add_argument('action', choices=['list', 'verify', 'import', 'download'])
    assets.add_argument('id', nargs='?')
    assets.add_argument('--root', default='artifacts')
    assets.add_argument('--manifest', type=Path)
    assets.add_argument('--file', type=Path)
    assets.add_argument('--filter', default='')
    bundle = sub.add_parser('bundles', help='Verify and install dependency-complete model/data archive bundles.')
    bundle.add_argument('action', choices=['list','verify','install','download'])
    bundle.add_argument('profile', nargs='?')
    bundle.add_argument('--root', type=Path, default=Path('artifacts'))
    bundle.add_argument('--bundle-dir', type=Path, default=Path('bundles'))
    bundle.add_argument('--manifest', type=Path)
    bundle.add_argument('--asset-manifest', type=Path)
    saved = sub.add_parser('reproduce', help='Recompute saved numerical results; this does not construct a new attack.')
    saved.add_argument('pathway', choices=['saved'])
    saved.add_argument('--data-root', type=Path, default=Path('data/saved'))
    saved.add_argument('--output', type=Path, required=True)
    saved.add_argument('--figures', action='store_true')
    smoke = sub.add_parser('smoke', help='Synthetic CPU functional test; no pretrained model or simulator.')
    smoke.add_argument('--output', type=Path, required=True)
    smoke.add_argument('--seed', type=int, default=7)
    cal = sub.add_parser('calibrate-detector', help='Freeze incoming-buffer CUSUM using disjoint saved clean Original residuals.')
    cal.add_argument('--statistics', type=Path, required=True)
    cal.add_argument('--threshold', type=Path, required=True)
    cal.add_argument('--family', choices=['tdmpc2','maniskill','graybox','dino-wm'], default='tdmpc2')
    cal.add_argument('--task', help='Task identity for the recorded statistics-cohort size and reference binding.')
    cal.add_argument('--output', type=Path, required=True)
    det = sub.add_parser('score-detector', help='Score every incoming trajectory using an already frozen CUSUM calibration.')
    for key in ['calibration','clean-test','poison','output']:
        det.add_argument('--'+key, type=Path, required=True)
    for name in ['run', 'prepare', 'stage1', 'select', 'construct', 'fit', 'evaluate', 'aggregate',
                 'collect-detector', 'detect-buffer']:
        x = sub.add_parser(name, help='Run the named scientific stage from an explicit family recipe.')
        x.add_argument('--config', type=Path, required=True)
        x.add_argument('--output', type=Path, required=True)
        x.add_argument('--artifact-root', type=Path, default=Path('artifacts'))
        x.add_argument('--manifest', type=Path)
        x.add_argument('--seed', type=int)
        x.add_argument('--task')
        x.add_argument('--alpha', type=float)
        x.add_argument('--poison-fraction', type=float)
        x.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
        x.add_argument('--dry-run', action='store_true', help='Resolve configuration only; not a simulator or model test.')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'assets':
            from .artifacts import AssetStore
            store = AssetStore(args.root, args.manifest)
            if args.action == 'list':
                result = [dict(id=k, available=store.destination(k).is_file(),
                               public_download=bool(v.get('urls')),
                               bytes=v.get('bytes', v.get('size_bytes')))
                          for k, v in sorted(store.entries.items()) if args.filter in k]
            else:
                if not args.id:
                    raise ValueError('An asset ID is required.')
                if args.action == 'import':
                    if not args.file:
                        raise ValueError('assets import requires --file PATH.')
                    result = store.import_file(args.id, args.file)
                elif args.action == 'verify':
                    result = store.verify(args.id)
                else:
                    result = store.download(args.id)
        elif args.command == 'bundles':
            from .bundles import BundleStore
            store=BundleStore(args.root,args.bundle_dir,args.manifest,args.asset_manifest)
            if args.action=='list':
                result=store.list_profiles()
            else:
                if not args.profile:
                    raise ValueError('A bundle profile is required. Use swaap bundles list.')
                result=getattr(store,args.action)(args.profile)
        elif args.command == 'reproduce':
            from .reproduce.saved import reproduce
            result = reproduce(args.data_root, args.output, figures=args.figures)
        elif args.command == 'smoke':
            from .smoke import run
            result = run(args.output, args.seed)
        elif args.command == 'calibrate-detector':
            from .evaluation.detector import calibrate
            result = calibrate(args.statistics,args.threshold,args.output,args.family,args.task)
        elif args.command == 'score-detector':
            from .evaluation.detector import score
            result = score(args.calibration,args.clean_test,args.poison,args.output)
        else:
            from .artifacts import AssetStore
            from .config import load_config, resolve_assets, create_output
            config = load_config(args.config, args.set)
            for key in ['seed', 'task', 'alpha', 'poison_fraction']:
                value = getattr(args, key)
                if value is not None:
                    config[key] = value
            family = config.get('family')
            aliases = {'tdmpc2': 'latent', 'maniskill': 'latent', 'maniskill2': 'latent',
                       'latent': 'latent', 'graybox': 'graybox', 'dino': 'dino', 'dino-wm': 'dino'}
            if family not in aliases:
                raise ValueError('Unknown family in configuration: ' + str(family))
            store = AssetStore(args.artifact_root, args.manifest)
            config, assets = resolve_assets(config, store)
            config['output_dir'] = str(args.output.expanduser().resolve())
            operation = 'prepare_data' if args.command == 'prepare' else args.command
            suffix='pipeline'
            if args.command in ('collect-detector','detect-buffer'):
                suffix='detector'
                operation='collect_original' if args.command=='collect-detector' else 'score_buffer'
            module_name = 'swaap.' + aliases[family] + '.'+suffix
            if args.command=='run':
                module_name='swaap.workflow'
                operation='fixed_target'
            if args.dry_run:
                result = dict(status='CONFIGURATION_ONLY_NOT_EXECUTED', entrypoint=module_name+'.'+operation,
                              configuration=config, assets=assets)
            else:
                module = importlib.import_module(module_name)
                function = getattr(module, operation, None)
                if function is None:
                    raise ValueError('This family does not define ' + operation + '; see its reproduction guide.')
                # The family owns creation of scientific output files; reserve only
                # the parent and reject an existing run before any model loads.
                output = args.output.expanduser().resolve()
                if output.exists():
                    raise ValueError('Output already exists; choose a new directory. No implicit resume.')
                output.parent.mkdir(parents=True, exist_ok=True)
                result = function(config)
                output.mkdir(parents=True, exist_ok=True)
                resolved = {k: v for k, v in config.items() if not k.startswith('_')}
                (output/'config.resolved.yaml').write_text(yaml.safe_dump(resolved, sort_keys=True))
                (output/'run.json').write_text(json.dumps(dict(stage=args.command, family=family,
                    package_version=__version__, result=result), indent=2, default=str)+'\n')
        print(json.dumps(result, indent=2, default=str, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, ModuleNotFoundError, ImportError) as error:
        print('swaap: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
