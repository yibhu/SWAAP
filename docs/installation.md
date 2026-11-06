# Installation

Use a separate environment for saved-result reproduction and each simulator
family. The package CLI itself requires only Python and PyYAML; it imports models
and simulators only when the requested command needs them.

## CPU tables and figures

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[saved,plots,test]'
swaap --help
```

The pinned plotting versions reproduce the numerical inputs and the supplied
plotting style. Font availability and PDF metadata can change output bytes across
systems. Saved inputs are verified independently of that rendering variation.

## DMControl, MyoSuite, and ManiSkill

The executed TD-MPC2 numerical runtime uses Python3.9.21 and
Torch2.6.0.dev20241112+cu124. Install its actual historical wheel, not a current
Torch nightly with a similar name. See [the latent-family guide](latent.md) for
complete commands using the resolved files:

| Family | Environment declaration | Resolved validation packages |
|---|---|---|
| DMControl | `environments/tdmpc2.yml` | `environments/tdmpc2-resolved.txt` |
| MyoSuite2.10.3 | `environments/tdmpc2-myosuite.yml` | `environments/tdmpc2-myosuite-resolved.txt` |
| Corrected ManiSkill2 | `environments/tdmpc2-maniskill.yml` | `environments/tdmpc2-maniskill-resolved.txt` |

The corrected ManiSkill experiments bind ManiSkill2 **0.5.3**, SAPIEN **2.2.2**,
NumPy1.23.5 and state-only observations. The old0.4.1/2.2.1 qualification is not
an interchangeable paper environment. The state-only adapter preserves native
reset/control/dynamics/reward code and rejects camera/renderer entry. Rendering
support is not needed and is not promised. A Vulkan loader library may be needed
by SAPIEN's binary import even when no Vulkan renderer is created.

MyoSuite needs its task assets; the asset guide and family guide give the exact
inputs. Retain its NumPy/MuJoCo versions in a separate environment. Do not install
the CPU plotting extras, which require NumPy2, into these environments.

MetaWorld and the raw gray-box route use the legacy MuJoCo210/MetaWorld binding.
Follow the exact installation and acquisition instructions in
[family workflows](families.md); current MetaWorld packages do not substitute for
that implementation. DMControl does not require this legacy binding.

## DINO-WM

Use Python3.11.13 on Linux and a CUDA12.8-compatible GPU/driver:

```bash
PYTHON_BIN=python3.11 bash environments/install-dino.sh .venv-dino
source .venv-dino/bin/activate
swaap --help
```

The installer downloads the recorded Torch2.9.0.dev20250723+cu128 and torchvision
wheel bodies, checks their exact sizes and SHA-256 values, installs the pinned
closure in `environments/dino-requirements.lock`, and runs `pip check`. It creates
a new environment and refuses to overwrite an existing one. The public observed
collection path needs neither the optional historical MP4 reader nor its
incompatible legacy decord wheel. The six PushT normalization tensors are
preserved verbatim in the code with their source identity.

## Assets and execution

Install the needed [asset profile](../assets/README.md) into a configurable
`artifacts/` directory. Install this source in each model environment with
`python -m pip install --no-deps .`. Run commands from any directory by supplying
absolute configuration, asset-root, and output paths. No private checkout,
notebook execution, scheduler, remote host, or existing environment name is
required.

The historical CUDA12.4 Torch build is not a supported Blackwell runtime. The
DINO CUDA12.8 runtime is separate. See [testing](testing.md) for the actual
hardware checks; using another runtime defines a separately labeled execution,
not an exact environment reproduction.
