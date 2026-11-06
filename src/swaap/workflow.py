"""A fixed-target attack and its matched Clean comparison, using ordinary files."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean


def fixed_target(config):
    """Construct, fit, reload and evaluate one bound latent TD-MPC2 attack.

    The selected target is an input. This function performs no target search,
    retry, selection by observed harm or automatic follow-on experiment.
    """
    if config['family'] not in ('tdmpc2','maniskill','latent'):
        raise ValueError('The fixed-target workflow uses latent TD-MPC2. Gray-box and DINO-WM have separate artifact workflows.')
    required = ['original', 'clean', 'reference', 'roots']
    # The two random-data comparators generate their own target field. Their
    # construction must not depend on an unrelated learned target checkpoint.
    if config.get('method', 'swaap') not in ('swaap_legacy_random', 'random500', 'clean'):
        required.append('target')
    for key in required:
        value=config.get('assets',{}).get(key)
        if not value or not Path(value).is_file():
            raise FileNotFoundError('Missing fixed-target input assets.'+key+'. Install the minimal bundle or bind the generated file.')
    root=Path(config['output_dir']).resolve()
    root.mkdir(parents=True,exist_ok=False)
    from .latent import pipeline
    def invoke(stage,name,changes):
        local=deepcopy(config);local['output_dir']=str(root/name)
        for key,value in changes.items():
            if key=='assets':local['assets'].update(value)
            else:local[key]=value
        return getattr(pipeline,stage)(local)
    construction=invoke('construct','construction',{})
    clean=invoke('fit','clean_fit',{'method':'clean','assets':{'data':config['assets']['clean']}})
    order=config['assets'].get('order') or str(root/'clean_fit/order.pt')
    poisoned=invoke('fit','attack_fit',{'assets':{'data':construction['files']['poison'],'order':order}})
    if not clean.get('order_sha256') or clean.get('order_sha256')!=poisoned.get('order_sha256'):
        raise ValueError('Clean and poison did not consume the same fitting permutations.')
    clean_eval=invoke('evaluate','clean_evaluation',{'method':'clean','assets':{'checkpoint':clean['checkpoint']}})
    attack_eval=invoke('evaluate','attack_evaluation',{'assets':{'checkpoint':poisoned['checkpoint']}})
    c,a=clean_eval['episodes'],attack_eval['episodes']
    identity=lambda rows:[(r['episode_index'],r['environment_seed'],r['planner_seed']) for r in rows]
    if not c or identity(c)!=identity(a):
        raise ValueError('Matched evaluation roots differ or the panel is empty.')
    clean_mean=mean(float(r['return']) for r in c)
    attack_mean=mean(float(r['return']) for r in a)
    result=dict(construction=construction,clean_fit=clean,attack_fit=poisoned,
        matched_clean=clean_eval,attack=attack_eval,target_selection_performed=False,
        clean_return=clean_mean,attack_return=attack_mean,matched_harm=clean_mean-attack_mean,
        relative_reduction_percent=100*(clean_mean-attack_mean)/clean_mean if clean_mean>0 else None,
        paired_harm_vector=[float(x['return'])-float(y['return']) for x,y in zip(c,a)],
        statistical_unit='one constructed attack and one matched fitting order; episodes are nested evaluations')
    with (root/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    return result
