"""The paper's independent unit is a Stage-1 run, not an episode or a fit."""
from collections import defaultdict
import math
from statistics import mean, stdev


def hierarchy(rows, value='value', task='task', run='stage1_run', fit='fit_order',
              expected_orders=None, expected_episodes=None):
    """Episode -> fitting-order mean -> Stage-1 mean -> task mean/sample SD.

    Missing observations are errors, never zeros. A run with no accepted target
    belongs in a separate selection-disposition record, not this numeric input.
    """
    grouped = defaultdict(list)
    seen_episodes = set()
    for row in rows:
        raw = row.get(value)
        if raw is None or raw == '':
            raise ValueError('A required observation is missing; do not impute it.')
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError('Non-finite observations are not valid measurements.')
        key = row[task], row[run], row[fit]
        if 'episode_index' in row:
            episode = key + (int(row['episode_index']),)
            if episode in seen_episodes:
                raise ValueError('Duplicate evaluation episode.')
            seen_episodes.add(episode)
        grouped[key].append(number)
    by_run = defaultdict(list)
    for key, values in grouped.items():
        if expected_episodes is not None and len(values) != expected_episodes:
            raise ValueError('Incomplete episode panel: ' + str(key))
        by_run[key[:2]].append(mean(values))
    by_task = defaultdict(list)
    for key, values in by_run.items():
        if expected_orders is not None and len(values) != expected_orders:
            raise ValueError('Incomplete fitting-order panel: ' + str(key))
        by_task[key[0]].append((key[1], mean(values)))
    return [dict(task=name, mean=mean(v for _, v in values),
                 sample_sd=stdev(v for _, v in values) if len(values) > 1 else None,
                 independent_stage1_runs=len(values),
                 stage1_means={run_id: value for run_id, value in values})
            for name, values in sorted(by_task.items())]


def buffer_hierarchy(rows, value='value'):
    """One score per fixed buffer; reject replication across victim-fit orders."""
    identities = set()
    data = []
    for row in rows:
        identity = row['task'], row['stage1_run']
        if identity in identities:
            raise ValueError('A fixed poison buffer cannot be counted once per fit.')
        identities.add(identity)
        data.append(dict(row, fit_order='one_fixed_buffer', value=row[value]))
    return hierarchy(data, expected_orders=1, expected_episodes=1)


def paired_reduction(clean, poison):
    if not math.isfinite(clean) or not math.isfinite(poison) or clean <= 0:
        raise ValueError('Percentage reduction requires a finite positive matched Clean return.')
    return 100.0 * (clean - poison) / clean
