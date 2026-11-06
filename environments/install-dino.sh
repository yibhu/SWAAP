#!/usr/bin/env bash
# Create an independent DINO environment; do not modify an existing environment.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3.11}
prefix=${1:-.venv-dino}
wheel_cache=${SWAAP_WHEEL_CACHE:-.swaap-wheel-cache}
if [[ -e "$prefix" ]]; then
  printf '%s\n' "Environment path already exists: $prefix" >&2
  exit 2
fi
"$python_bin" -c 'import sys; assert sys.version_info[:3]==(3,11,13), "Use Python3.11.13 (for example conda create -n swaap-dino-python python=3.11.13)."'
"$python_bin" -m venv --copies "$prefix"
"$prefix/bin/python" - "$script_dir/dino-wheels.json" "$wheel_cache" <<'PY'
import hashlib,json,sys,urllib.request
from pathlib import Path
records=json.loads(Path(sys.argv[1]).read_text());cache=Path(sys.argv[2]);cache.mkdir(parents=True,exist_ok=True)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
for row in records:
 target=cache/row['filename']
 if not target.exists():
  temporary=target.with_suffix('.partial')
  with urllib.request.urlopen(row['url'],timeout=60) as response,temporary.open('wb') as out:
   for block in iter(lambda:response.read(8*1024*1024),b''):out.write(block)
  temporary.rename(target)
 if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:
  raise SystemExit('Wheel checksum/size mismatch: '+str(target))
 print(target)
PY
"$prefix/bin/python" -m pip install --no-deps "$wheel_cache/torch-2.9.0.dev20250723+cu128-cp311-cp311-manylinux_2_28_x86_64.whl" "$wheel_cache/torchvision-0.24.0.dev20250723+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
"$prefix/bin/python" -m pip install -r "$script_dir/dino-requirements.lock"
"$prefix/bin/python" -m pip install --no-deps --no-build-isolation "$script_dir/.."
"$prefix/bin/python" -m pip check
printf '%s\n' "DINO environment installed at $prefix" "Activate with: source $prefix/bin/activate"
