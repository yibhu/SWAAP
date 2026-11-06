#!/usr/bin/env python3
"""Check public configuration/asset dependencies without loading a model.

This is a static acceptance check, not a claim that every experiment has run.
"""
import argparse
import ast
import csv
import json
from pathlib import Path

import yaml
from swaap.artifacts import AssetStore
from swaap.cli import parser as cli_parser


def check(root):
    root = Path(root).resolve()
    store = AssetStore(root / 'artifacts', root / 'assets/asset_manifest.json')
    ids, local_files, configs = set(), set(), []
    def visit(item, base):
        if isinstance(item, dict):
            for value in item.values():
                visit(value, base)
        elif isinstance(item, list):
            for value in item:
                visit(value, base)
        elif isinstance(item, str) and item.startswith('asset:'):
            store.entry(item[6:])
            ids.add(item[6:])
        elif isinstance(item, str) and item.startswith('config:'):
            path = (base / item[7:]).resolve()
            if not path.is_file():
                raise ValueError('Missing configuration-local input: ' + str(path))
            local_files.add(path.relative_to(root).as_posix())
    families = {'tdmpc2': 'latent', 'maniskill': 'latent', 'graybox': 'graybox', 'dino': 'dino'}
    for path in sorted((root / 'configs').rglob('*.yaml')):
        cfg = yaml.safe_load(path.read_text())
        if not isinstance(cfg, dict):
            raise ValueError('Recipe must be a mapping: ' + str(path))
        visit(cfg.get('assets', {}), path.parent)
        if 'assets' in cfg:
            family = families.get(cfg.get('family'))
            if family is None:
                raise ValueError('Unknown recipe family: ' + str(path))
            module = ast.parse((root / f'src/swaap/{family}/pipeline.py').read_text())
            names = {x.name for x in module.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
            required = {'prepare_data', 'stage1', 'select', 'construct', 'fit', 'evaluate', 'aggregate'}
            if not required.issubset(names):
                raise ValueError(f'{family} lacks named entry points: {required - names}')
        configs.append(path.relative_to(root).as_posix())
    rows = list(csv.DictReader((root / 'provenance/PAPER_TO_CODE.csv').open()))
    import shlex
    commands = []
    for row in rows:
        for command in row.get('artifact_command_sequence', '').split(';'):
            tokens = shlex.split(command)
            if not tokens:
                continue
            if tokens[0] == 'swaap':
                cli_parser().parse_args(tokens[1:])
                commands.append(command.strip())
            elif tokens[:2] == ['python', 'scripts/bind_order.py']:
                if not (root / tokens[1]).is_file():
                    raise ValueError('Missing order-binding script.')
                commands.append(command.strip())
            else:
                raise ValueError('Non-executable artifact mapping: ' + command)
        for asset_id in filter(None, row.get('required_asset_ids', '').split(';')):
            store.entry(asset_id)
    return {'status': 'STATIC_CONFIGURATION_AND_COMMAND_GRAPH_PASS',
            'configurations': len(configs), 'referenced_asset_ids': len(ids),
            'local_configuration_inputs': len(local_files), 'paper_rows': len(rows),
            'parsed_artifact_commands': len(commands),
            'scope': 'schema, known asset IDs, local input existence, callable definitions and CLI argument parsing; no model, installation or efficacy validation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = check(args.root)
    if args.output:
        with args.output.open('x') as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
