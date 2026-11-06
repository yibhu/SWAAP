"""Bounded synthetic CPU exercise of the scientific kernel, not an RL result."""
from pathlib import Path


def run(output, seed=7):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new directory.')
    from .latent.pipeline import smoke
    return smoke(dict(output_dir=str(output),seed=seed))
