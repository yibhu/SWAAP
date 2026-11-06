#!/usr/bin/env python3
"""Declare disjoint roots for a new latent-family detector/data collection.

This does not recover historical roots or run any model. The versioned seed
derivation is part of a newly declared stochastic experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

from swaap.artifacts import AssetStore

DMC = {'humanoid-walk', 'humanoid-run', 'dog-walk', 'dog-run', 'cheetah-run'}
OTHER = {'mw-push', 'mw-soccer', 'myo-pen-twirl-hard', 'myo-reach-hard'}
MANISKILL = {'lift-cube', 'pick-cube', 'stack-cube'}


def panels(task, seed, original_sha):
    if task not in DMC | OTHER | MANISKILL:
        raise ValueError('Unknown latent-family task.')
    if len(original_sha) != 64 or any(x not in '0123456789abcdef' for x in original_sha):
        raise ValueError('An exact Original SHA-256 is required.')
    statistics = 500 if task in MANISKILL | {'mw-push', 'humanoid-walk'} else 300
    counts = {'statistics': statistics, 'threshold': 500, 'clean_test': 1000,
              'incoming': 10 if task in DMC else 50,
              'reference': 100 if task in DMC else 500}
    used = set()
    used_environment, used_planner = set(), set()
    result = {}
    for role, count in counts.items():
        roots = []
        for index in range(count):
            prefix = f'swaap-detector-roots-v1:{seed}:{task}:{role}:{index}'
            pair = tuple(int.from_bytes(hashlib.sha256((prefix + ':' + stream).encode()).digest()[:4], 'big')
                         for stream in ('environment', 'planner'))
            if pair in used or pair[0] in used_environment or pair[1] in used_planner:
                raise ValueError('Seed collision; declare a different experiment seed.')
            used.add(pair)
            used_environment.add(pair[0])
            used_planner.add(pair[1])
            roots.append({'episode_index': index, 'environment_seed': pair[0], 'planner_seed': pair[1]})
        result[role] = {'schema_version': 1, 'task': task, 'role': role,
                        'original_sha256': original_sha, 'seed': seed,
                        'seed_derivation': 'swaap-detector-roots-v1/SHA256 first32 big-endian',
                        'provenance': 'new stochastic collection; not historical paper root recovery',
                        'roots': roots}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--original', required=True, help='Original asset ID, without asset: prefix.')
    parser.add_argument('--artifact-root', default='artifacts')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = AssetStore(args.artifact_root, args.manifest).verify(args.original)
    values = panels(args.task, args.seed, original['sha256'])
    args.output.mkdir(parents=True, exist_ok=False)
    for role, value in values.items():
        (args.output / (role + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    collection = {'schema_version': 1, 'task': args.task,
                  'original_sha256': original['sha256'],
                  'clean_roots': values['incoming']['roots'],
                  'reference_roots': values['reference']['roots'],
                  'provenance': values['incoming']['provenance']}
    (args.output / 'collection.json').write_text(json.dumps(collection, indent=2) + '\n')
    print(json.dumps({'output': str(args.output.resolve()),
                      'trajectory_counts': {role: len(value['roots']) for role, value in values.items()}}))


if __name__ == '__main__':
    main()
