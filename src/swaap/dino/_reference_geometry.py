import torch
def rms_regularizer(clean, prediction, editable):
    assert clean.shape == prediction.shape == editable.shape and clean.ndim == 3
    rc = (clean - prediction).square().flatten(1).mean(1).sqrt()
    rp = (editable - prediction).square().flatten(1).mean(1).sqrt()
    return (rp - rc).square().mean(), rc, rp

def cosine_vector(target, gradient):
    variable = gradient.detach().requires_grad_(True)
    assert bool(torch.isfinite(variable).all()) and float(variable.norm()) > 0
    cosine = torch.dot(target, variable) / (target.norm() * variable.norm())
    similarity = 1 - cosine
    vector = torch.autograd.grad(similarity, variable)[0].detach()
    return similarity.detach(), cosine.detach(), vector

def project(proposal, clean, epsilon=1.):
    """Canonical clean-centred box, including the BF16-consumed label."""
    low,high=clean.double()-epsilon,clean.double()+epsilon
    lo,hi=low.float(),high.float()
    lo=torch.where(lo.double()<low,torch.nextafter(lo,torch.full_like(lo,float('inf'))),lo)
    hi=torch.where(hi.double()>high,torch.nextafter(hi,torch.full_like(hi,-float('inf'))),hi)
    value=torch.maximum(torch.minimum(proposal.float(),hi),lo)
    lb,hb=lo.bfloat16(),hi.bfloat16()
    lb=torch.where(lb.double()<low,torch.nextafter(lb,torch.full_like(lb,float('inf'))),lb)
    hb=torch.where(hb.double()>high,torch.nextafter(hb,torch.full_like(hb,-float('inf'))),hb)
    assert bool((lb<=hb).all())
    value=torch.where(value.bfloat16().double()<low,lb.float(),value)
    value=torch.where(value.bfloat16().double()>high,hb.float(),value)
    return value
