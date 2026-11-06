import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from swaap.evaluation.detector import calibrate, score, scores

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_float64_scoring_boundary_retains_stored_statistic_dtype():
    # The canonical scorer widens before subtraction, even though collection
    # and robust center/scale fitting retain the stored float32 values.
    residuals = np.array([[1.25, 1.5]], dtype=np.float32)
    median = 1.00000003
    actual = scores({'residuals': residuals}, median, .01)
    expected = (1.25 - median - .06) + (1.5 - median - .06)
    assert actual == pytest.approx([expected], rel=0, abs=1e-15)
    assert residuals.dtype == np.float32


def test_new_root_derivation_is_deterministic_and_disjoint():
    spec = importlib.util.spec_from_file_location('make_roots', ROOT / 'scripts/make_detector_roots.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = module.panels('dog-run', 17, 'a' * 64)
    assert first == module.panels('dog-run', 17, 'a' * 64)
    assert {role: len(value['roots']) for role, value in first.items()} == {
        'statistics': 300, 'threshold': 500, 'clean_test': 1000, 'incoming': 10, 'reference': 100}
    pairs = [(r['environment_seed'], r['planner_seed']) for p in first.values() for r in p['roots']]
    assert len(pairs) == len(set(pairs))
    assert len({p[0] for p in pairs}) == len(pairs)
    assert len({p[1] for p in pairs}) == len(pairs)
    assert module.panels('lift-cube', 17, 'a' * 64)['statistics']['roots'] != first['statistics']['roots']
    assert len(module.panels('humanoid-walk', 17, 'a' * 64)['statistics']['roots']) == 500


def test_task_statistics_count_and_reference_metadata_are_enforced(tmp_path):
    def cohort(name, count, task='dog-run', **metadata):
        path = tmp_path / (name + '.npz')
        np.savez(path, residuals=np.ones((count, 3), np.float32),
                 trajectory_ids=np.array([f'{name}:{i}' for i in range(count)]),
                 original_sha256=np.array('a' * 64), task=np.array(task),
                 **{key: np.array(value) for key, value in metadata.items()})
        return path
    statistics = cohort('statistics', 300)
    threshold = cohort('threshold', 500)
    output = tmp_path / 'calibration.json'
    calibrate(statistics, threshold, output, 'tdmpc2', 'dog-run')
    assert json.loads(output.read_text())['task'] == 'dog-run'
    with pytest.raises(ValueError, match='500 statistics'):
        calibrate(cohort('hw-stat', 300, 'humanoid-walk'),
                  cohort('hw-threshold', 500, 'humanoid-walk'), tmp_path / 'too-small.json', 'tdmpc2')
    with pytest.raises(ValueError, match='threshold trajectories'):
        calibrate(statistics, cohort('too-small-threshold', 499), tmp_path / 'too-small-threshold.json', 'tdmpc2')
    with pytest.raises(ValueError, match='reference metadata'):
        score(output, cohort('test', 1000), cohort('wrong-task', 10, 'humanoid-walk'), tmp_path / 'scores.json')
