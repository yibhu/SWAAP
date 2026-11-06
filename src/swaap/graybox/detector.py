"""Public incoming-buffer detector entry points; Original source/victim stay distinct."""
from .detection import collect as collect_original


def score_buffer(config):
    from .detection import incoming
    from .pipeline import asset
    from swaap.evaluation.detector import score
    from swaap.latent.io import output,write_json
    root=output(config);inputs=incoming(config)
    result=score(asset(config,'calibration'),asset(config,'clean_test'),inputs['victim']['poison']['residuals'],root/'cusum.json')
    result['incoming_data_deviation']=inputs
    write_json(root/'detector.json',result);return result
