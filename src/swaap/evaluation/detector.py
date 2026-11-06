"""Explicit calibration/scoring of saved incoming-buffer Original residuals.

Inputs contain raw (unnormalized) residual sequences, never fitted-model rollout
errors. Model execution and clean trajectory collection are separate stages.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .cusum import robust_stats, cusum_score, split_conformal_threshold, evaluate_score_samples


def load(path):
    path=Path(path)
    with np.load(path,allow_pickle=False) as a:
        residuals=a['residuals'].copy()
        ids=a['trajectory_ids'].astype(str).tolist()
        reference=str(a['original_sha256'].item())
        metadata={key:str(a[key].item()) for key in
                  ('task','reference_domain','reference_task_index','residual_precision') if key in a.files}
    if residuals.ndim!=2 or not np.isfinite(residuals).all() or (residuals<0).any():
        raise ValueError('Expected finite nonnegative raw-L2 residuals [trajectories, steps].')
    if len(ids)!=len(residuals) or len(set(ids))!=len(ids):
        raise ValueError('Trajectory IDs must be unique and align with residuals.')
    if len(reference)!=64 or any(c not in '0123456789abcdef' for c in reference):
        raise ValueError('A frozen Original SHA-256 is required.')
    return dict(residuals=residuals,ids=ids,original=reference,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=metadata)


def disjoint(*cohorts):
    seen=set()
    for ids in cohorts:
        if seen.intersection(ids):
            raise ValueError('Statistics, threshold, clean-test and incoming roles must be disjoint.')
        seen.update(ids)


def scores(data, median, mad):
    # Canonical residual_cusum_protocol_v4.scores_from_residuals converts only
    # the scoring matrix to float64. Center/scale fitting above keeps the
    # stored dtype, as in the retained gray-box and DINO adapters.
    traces = np.asarray(data['residuals'], dtype=float)
    return [cusum_score(x,median=float(median),mad=float(mad),drift_factor=6.) for x in traces]


def calibrate(statistics, threshold, output, family='tdmpc2', task=None):
    stat,cal=load(statistics),load(threshold)
    disjoint(stat['ids'],cal['ids'])
    if stat['original']!=cal['original']:
        raise ValueError('Calibration inputs use different Original models.')
    for key in set(stat['metadata'])|set(cal['metadata']):
        if stat['metadata'].get(key)!=cal['metadata'].get(key):
            raise ValueError('Calibration reference metadata differs: '+key)
    if task and stat['metadata'].get('task',task)!=task:
        raise ValueError('Declared task differs from the residual reference task.')
    task=task or stat['metadata'].get('task')
    statistics_minimum={'dino-wm':200,'graybox':300}.get(family,500)
    if family=='tdmpc2' and task:
        # Source collect_residual_cusum_v4.py: only these two statistics banks
        # contain500 trajectories; the other seven use300. Thresholds remain500.
        known={'mw-push','humanoid-walk','humanoid-run','dog-walk','dog-run',
               'cheetah-run','myo-pen-twirl-hard','myo-reach-hard','mw-soccer'}
        if task not in known:raise ValueError('Unknown TD-MPC2 calibration task.')
        statistics_minimum=500 if task in ('mw-push','humanoid-walk') else 300
    threshold_minimum=200 if family=='dino-wm' else 500
    if len(stat['ids'])<statistics_minimum or len(cal['ids'])<threshold_minimum:
        raise ValueError(f'{family} needs at least {statistics_minimum} statistics and {threshold_minimum} threshold trajectories.')
    median,mad=robust_stats(stat['residuals'].reshape(-1))
    values=scores(cal,median,mad);thresholds={}
    for fpr in [.05,.10]:
        value,rank=split_conformal_threshold(values,fpr)
        thresholds[f'{fpr:.2f}']=dict(threshold=None if np.isinf(value) else value,
              threshold_is_positive_infinity=bool(np.isinf(value)),rank_one_based=rank)
    record=dict(family=family,task=task,reference_metadata=stat['metadata'],
         domain='incoming buffer raw Original residual L2; no normalization',
         original_sha256=stat['original'],median=median,mad=mad,drift_factor=6.,
         statistics_ids=stat['ids'],threshold_ids=cal['ids'],thresholds=thresholds,
         statistics_sha256=stat['sha256'],threshold_sha256=cal['sha256'],alarm_rule='score > threshold')
    path=Path(output)
    with path.open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
    return dict(status='FROZEN',path=str(path),statistics_trajectories=len(stat['ids']),
                threshold_trajectories=len(cal['ids']))


def score(calibration, clean_test, poison, output):
    lock=json.loads(Path(calibration).read_text());clean,incoming=load(clean_test),load(poison)
    disjoint(lock['statistics_ids'],lock['threshold_ids'],clean['ids'],incoming['ids'])
    if {clean['original'],incoming['original'],lock['original_sha256']}!={lock['original_sha256']}:
        raise ValueError('Scoring must use the same frozen Original as calibration.')
    for key,value in lock.get('reference_metadata',{}).items():
        if clean['metadata'].get(key)!=value or incoming['metadata'].get(key)!=value:
            raise ValueError('Scoring reference metadata differs from calibration: '+key)
    minimum=400 if lock['family']=='dino-wm' else 1000
    if len(clean['ids'])<minimum:
        raise ValueError(f'Independent clean test needs at least {minimum} trajectories.')
    result=evaluate_score_samples(scores(clean,lock['median'],lock['mad']),
          scores(incoming,lock['median'],lock['mad']),lock['thresholds'],
          bootstrap_seed=20261104 if lock['family']=='dino-wm' else 20260925)
    result.update(calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
                  poison_sha256=incoming['sha256'],clean_test_sha256=clean['sha256'],
                  domain=lock['domain'],trajectory_ids=incoming['ids'])
    with Path(output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    return result
