"""Executed canonical FP32 incoming-data metrics, separate from BF16 construction.

Pure saved-array operations. No model or environment execution is performed.
"""
import numpy as np
import torch

def _tensor32(value):
    value = value.detach().cpu() if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
    assert value.dtype == torch.float32, 'Canonical FP32 inputs; no silent BF16/FP64 conversion'
    assert bool(value.isfinite().all())
    return value

def _state_fields(successor, prediction):
    successor, prediction = _tensor32(successor), _tensor32(prediction)
    assert successor.shape == prediction.shape and successor.ndim == 3
    assert successor.shape[1:] == (196, 404)
    return successor[..., :394], prediction[..., :394]

def _record_ids(origins, windows):
    assert windows % 8 == 0 and len(origins) == windows
    ids = []
    for start in range(0, windows, 8):
        rows = origins[start:start + 8]
        index = int(rows[0]['record_index'])
        assert all(int(row['record_index']) == index for row in rows)
        assert [int(row['local_window']) for row in rows] == list(range(8))
        for local, row in enumerate(rows):
            if 'last_current_grouped_step' in row:
                assert row['last_current_grouped_step'] == local + 2
            if 'successor_grouped_step' in row:
                assert row['successor_grouped_step'] == local + 3
        ids.append(index)
    assert len(set(ids)) == len(ids), 'One contiguous eight-window block per trajectory'
    return ids

def residuals_from_predictions(successor, prediction, origins):
    y, p = _state_fields(successor, prediction)
    ids = _record_ids(origins, len(y))
    residual = (y - p).flatten(1).norm(p=2, dim=1)
    return residual.numpy().reshape(len(ids), 8).copy(), ids

def normalized_deviation(successor, prediction, selected=None):
    y, p = _state_fields(successor, prediction)
    if selected is not None:
        indices = torch.as_tensor(selected, dtype=torch.long).cpu()
        assert indices.shape == (32,) and indices.unique().numel() == 32
        assert bool(((indices >= 0) & (indices < len(y))).all())
        y, p = y[indices], p[indices]
    values = (y - p).flatten(1).norm(p=2, dim=1) / (p.flatten(1).norm(p=2, dim=1) + 1e-8)
    return dict(windows=len(values), mean=float(values.mean()), per_window=values.tolist(),
        domain='Canonical FP32 labels/reference predictions; state196x394, no BF16 cast',
        definition='mean_i ||y_i-p_i||2/(||p_i||2+1e-8); separate from raw CUSUM residuals')
