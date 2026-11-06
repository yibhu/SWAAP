"""Configuration resolution without a scientific runtime or a private checkout."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml


def load_config(path, overrides=()):
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('A configuration must contain a YAML mapping.')
    config = deepcopy(config)
    for expression in overrides:
        if '=' not in expression:
            raise ValueError('--set requires key=value')
        key, raw = expression.split('=', 1)
        keys = key.split('.')
        if not all(keys):
            raise ValueError('Empty configuration key.')
        item = config
        for name in keys[:-1]:
            item = item.setdefault(name, {})
            if not isinstance(item, dict):
                raise ValueError('Override enters a non-mapping: ' + key)
        item[keys[-1]] = yaml.safe_load(raw)
    config['_config_file'] = str(path)
    config['_config_dir'] = str(path.parent)
    return config


def resolve_assets(config, store):
    """Resolve IDs, verify present files, and leave absent assets explicit."""
    statuses = []
    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str) and value.startswith('asset:'):
            asset_id = value[6:]
            destination = store.destination(asset_id)
            if destination.exists():
                statuses.append(store.verify(asset_id))
            else:
                statuses.append(dict(id=asset_id, path=str(destination), status='MISSING'))
            return str(destination)
        if isinstance(value, str) and value.startswith('config:'):
            return str((Path(config['_config_dir'])/value[7:]).resolve())
        return value
    result = deepcopy(config)
    result['assets'] = visit(result.get('assets', {}))
    return result, statuses


def create_output(path):
    path = Path(path).expanduser().resolve()
    # Outputs are never overwritten or silently resumed.
    path.mkdir(parents=True, exist_ok=False)
    return path
