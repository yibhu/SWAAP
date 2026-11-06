"""Small ordinary-file utilities, without execution infrastructure."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    require(value is not None and type(value) is not bool, 'Missing/boolean scientific scalar')
    value = float(value)
    require(math.isfinite(value), 'Nonfinite scientific scalar')
    return value


def asset(config, name, *, required=True):
    value = config.get('assets', {}).get(name)
    if not value:
        if not required:
            return None
        family = config.get('family', 'tdmpc2')
        raise FileNotFoundError(f"Provide assets.{name}, for example asset:{family}.{config.get('task', '<task>')}.{name}")
    path = Path(value).expanduser()
    require(path.is_file(), f'Missing assets.{name}: {value}. Import the exact asset using the asset manifest; no substitute is loaded.')
    return path.resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return str(path.resolve())


def output(config):
    root = Path(config['output_dir']).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_tensor(path, value):
    import torch
    path = Path(path)
    require(not path.exists(), f'Refusing to overwrite an existing scientific artifact: {path}')
    torch.save(value, path)
    return str(path.resolve())


def load_buffer(path, *, rows=None):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    require(isinstance(value, (list, tuple)) and len(value) == 5 and all(isinstance(x, torch.Tensor) for x in value),
            'A five-column tensor tuple (state, action, reward, next_state, done) is required')
    require(len({len(x) for x in value}) == 1 and len(value[0]) > 0, 'Invalid buffer dimensions')
    require(rows is None or len(value[0]) == rows, f'Expected {rows} rows')
    require(all(bool(torch.isfinite(x).all()) for x in value), 'Nonfinite buffer')
    return tuple(value)


def load_support(path,*,rows,count):
    """Read frozen ordered indices from Torch or a JSON integer list."""
    import torch
    path=Path(path)
    if path.suffix.lower()=='.json':
        values=read_json(path)
        require(isinstance(values,list)and all(type(v)is int for v in values),
                'JSON support must be an ordered integer list, not scores or a new ranking')
        value=torch.tensor(values,dtype=torch.int64)
    else:
        value=torch.load(path,map_location='cpu',weights_only=False)
    require(isinstance(value,torch.Tensor)and value.dtype==torch.int64 and value.ndim==1 and value.numel()==count,
            'Frozen support requires the exact ordered int64 count')
    require(len(set(value.tolist()))==count and bool(((value>=0)&(value<rows)).all()),
            'Frozen support indices are duplicated or outside the bank')
    return value.cpu()
