#!/usr/bin/env bash
# Add the legacy simulator layer to a separately created TD-MPC2 environment.
# Use a new runtime prefix; this script never modifies system libraries.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python}
runtime_prefix=${1:?Usage: bash environments/install-graybox.sh /absolute/new/runtime-prefix}
if [[ -e "$runtime_prefix" ]]; then
  printf '%s\n' "Runtime prefix already exists: $runtime_prefix" >&2
  exit 2
fi
"$python_bin" - <<'PY'
import sys,torch
assert sys.version_info[:3]==(3,9,21), 'Use the Python3.9.21 TD-MPC2 base environment.'
assert torch.__version__=='2.6.0.dev20241112+cu124', 'Install the recorded CUDA12.4 Torch wheel first.'
assert not torch.cuda.is_initialized()
PY
command -v gcc >/dev/null || { printf '%s\n' 'Install a C compiler (Ubuntu: sudo apt install build-essential).' >&2; exit 2; }
command -v dpkg-deb >/dev/null || { printf '%s\n' 'This user-prefix installer targets Ubuntu22.04 x86_64 and needs dpkg-deb.' >&2; exit 2; }
"$python_bin" - "$script_dir/graybox-native-assets.json" "$runtime_prefix" <<'PY'
import hashlib,json,platform,subprocess,sys,tarfile,urllib.request
from pathlib import Path
assert platform.system()=='Linux' and platform.machine()=='x86_64', 'Legacy profile targets Linux x86_64.'
root=Path(sys.argv[2]).resolve();root.mkdir();cache=root/'downloads';cache.mkdir()
for row in json.loads(Path(sys.argv[1]).read_text()):
    archive=cache/row['filename']
    with urllib.request.urlopen(row['url'],timeout=60) as response,archive.open('xb') as out:
        for block in iter(lambda:response.read(1024*1024),b''):out.write(block)
    assert archive.stat().st_size==row['bytes'], 'Incorrect download size: '+row['filename']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['sha256'], 'Download hash mismatch: '+row['filename']
    if row['format']=='deb':
        subprocess.run(['dpkg-deb','--extract',str(archive),str(root/'osmesa')],check=True)
    else:
        with tarfile.open(archive) as tar:
            assert all(not m.name.startswith('/') and '..' not in Path(m.name).parts for m in tar.getmembers())
            tar.extractall(root)
print('Verified MuJoCo210 and native development packages:',root)
PY
"$python_bin" -m pip install -r "$script_dir/graybox-requirements.lock"
"$python_bin" -m pip install --no-deps 'git+https://github.com/rlworkgroup/metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb'
python_prefix=$("$python_bin" -c 'import sys; print(sys.prefix)')
runtime_prefix=$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$runtime_prefix")
# Keep all paths explicit. The GL/OSMesa libraries are build dependencies; no
# camera/render observation or env.render call is used by these experiments.
{
  printf 'export PATH=%q/bin:"$PATH"\n' "$python_prefix"
  printf 'export MUJOCO_PY_MUJOCO_PATH=%q/mujoco210\n' "$runtime_prefix"
  printf 'export MUJOCO_PY_FORCE_CPU=1\nexport MUJOCO_GL=egl\n'
  printf 'export LD_LIBRARY_PATH=%q/osmesa/usr/lib/x86_64-linux-gnu:%q/mujoco210/bin:%q/lib:"${LD_LIBRARY_PATH:-}"\n' "$runtime_prefix" "$runtime_prefix" "$python_prefix"
  printf 'export CPATH=%q/osmesa/usr/include:%q/include:"${CPATH:-}"\n' "$runtime_prefix" "$python_prefix"
  printf 'export LIBRARY_PATH=%q/osmesa/usr/lib/x86_64-linux-gnu:%q/lib:"${LIBRARY_PATH:-}"\n' "$runtime_prefix" "$python_prefix"
} > "$runtime_prefix/activate-graybox.sh"
source "$runtime_prefix/activate-graybox.sh"
"$python_bin" - <<'PY'
import torch
assert not torch.cuda.is_initialized()
import mujoco_py,metaworld
assert mujoco_py.__version__=='2.1.2.14'
assert not torch.cuda.is_initialized()
print('Legacy simulator imports passed without CUDA initialization.')
PY
"$python_bin" -m pip install --no-deps --no-build-isolation "$script_dir/.."
"$python_bin" -m pip check
printf '%s\n' "Source the runtime variables before a new session:" "source $runtime_prefix/activate-graybox.sh"
