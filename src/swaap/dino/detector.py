"""DINO FP32 incoming-buffer collection and fixed-calibration scoring."""
from .detection import collect as collect_original


def score_buffer(config):
    from .pipeline import _incoming_metrics
    from swaap.evaluation.detector import score
    from swaap.latent.io import asset,output,write_json,require
    root=output(config);inputs=_incoming_metrics(config)
    require('raw_residual_files' in inputs,'Exact saved window_origins are required for DINO incoming CUSUM')
    result=score(asset(config,'calibration'),asset(config,'clean_test'),inputs['raw_residual_files']['poison'],root/'cusum.json')
    result['incoming_data_deviation']=inputs
    write_json(root/'detector.json',result);return result
