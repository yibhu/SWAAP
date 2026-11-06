"""Unchanged executed target loader used for CPU equivalence fixtures."""
from copy import deepcopy

def target_model(root,item,original,torch):
    payload=torch.load(root/item['checkpoint'],map_location='cpu',weights_only=False)
    if 'model'in payload:
        state=payload['model'];source=original.model.state_dict()
        for tp,op in [('psi_0.','_dynamics.'),('_encoder.','_encoder.')]:
            a={k[len(tp):]:v for k,v in state.items()if k.startswith(tp)and isinstance(v,torch.Tensor)}
            b={k[len(op):]:v.cpu()for k,v in source.items()if k.startswith(op)and isinstance(v,torch.Tensor)}
            assert a.keys()==b.keys()and all(torch.equal(a[k],b[k])for k in a),item['id']
        dynamics={k[len('psi.'):]:v for k,v in state.items()if k.startswith('psi.')}
    else:
        # Native Stage-1 saves its target dynamics and optimized actor separately.
        # Only psi_k is consumed. Actor/reward/Q and the prospective source encoder
        # stay Original. Missing historical encoder bytes remain a provenance limit.
        assert item['checkpoint_format']=='psi_k_theta_k' and 'psi_k'in payload
        dynamics=payload['psi_k']
    target=deepcopy(original.model._dynamics)
    target.load_state_dict(dynamics,strict=True)
    return target.eval().requires_grad_(False)
