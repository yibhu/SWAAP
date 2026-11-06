"""Frozen DINO-WM family science: BF16 chunks, FP32 masters, full404 loss.

The editable 32 windows are restricted to the first80 of320 rows. Their action
features394:404 and every unselected field stay fixed. This is not the latent
TD-MPC2 constructor.
"""
from copy import deepcopy
import hashlib
import torch
def latent_next(model, state, action):
    """Identical latent branch to ThetaPsi.wm_next; unused decoding omitted."""
    current = model.replace_actions_from_z(state[:, -1:].clone(), action[:, None])
    history = torch.cat([state[:, :-1], current], dim=1)
    return model.predict(history[:, -3:])[:, -1]

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

class GradientEngine:
    def __init__(self, model, train, reference, target_reference_labels, support,
                 predictions, canonical_clean, chunk_size=4):
        assert chunk_size == 4
        assert train[0].shape == (320, 3, 196, 404) and train[3].shape == (320, 196, 404)
        assert reference[0].shape == (80, 3, 196, 404)
        assert target_reference_labels.shape == reference[3].shape == (80, 196, 404)
        assert support.shape == (32,) and support.unique().numel() == 32
        assert bool(((support >= 0) & (support < 80)).all())
        assert predictions.shape == canonical_clean.shape == train[3].shape
        assert not model.predictor.training, 'Construction uses frozen eval/dropout-off semantics'
        self.model, self.train, self.reference = model, train, reference
        self.params = tuple(model.predictor.parameters())
        assert self.params and all(p.requires_grad and p.dtype == torch.bfloat16 for p in self.params)
        self.device = self.params[0].device
        self.support = support.detach().to(self.device, dtype=torch.long).clone()
        self.ids = self.support.cpu().tolist()
        self.chunk_size = chunk_size
        self.target_labels = target_reference_labels.detach().to(self.device).bfloat16()
        assert torch.equal(self.target_labels[..., 394:], reference[3][..., 394:].bfloat16())
        self.clean = canonical_clean[self.support.to(canonical_clean.device), :, :394].detach().to(self.device).float().clone()
        self.predictions = predictions[self.support.to(predictions.device), :, :394].detach().to(self.device).float().clone()
        self.tail = train[3][self.support, :, 394:].detach().float().clone()
        self.graphs = []
        self.gt = self.fixed = None

    def _flat(self, prediction, labels, create_graph=False):
        loss = torch.nn.functional.mse_loss(prediction, labels.bfloat16(), reduction='mean')
        grads = torch.autograd.grad(loss, self.params, create_graph=create_graph, retain_graph=True)
        return torch.cat([value.reshape(-1).float() for value in grads]), loss.detach()

    def _mean(self, data, labels, ids, denominator):
        result = torch.zeros(sum(p.numel() for p in self.params), device=self.device, dtype=torch.float32)
        for offset in range(0, len(ids), self.chunk_size):
            chunk = torch.tensor(ids[offset:offset + self.chunk_size], device=self.device)
            prediction = latent_next(self.model, data[0][chunk], data[1][chunk])
            gradient, _ = self._flat(prediction, labels[chunk])
            result.add_(gradient, alpha=len(chunk) / denominator)
            del prediction, gradient
        return result

    def cache(self, saved=None):
        assert not self.graphs, 'Do not duplicate an existing selected graph cache'
        if saved is None:
            self.gt = self._mean(self.reference, self.target_labels, list(range(80)), 80)
            selected = set(self.ids)
            self.fixed = self._mean(self.train, self.train[3],
                                    [i for i in range(320) if i not in selected], 320)
        else:
            assert saved['schema'] == 'FAITHFUL_ORDINARY_SWAAP_CHUNK4_V1'
            assert saved['train_rows'] == 320 and saved['reference_rows'] == 80
            assert saved['chunk_size'] == self.chunk_size
            assert saved['support'].tolist() == self.ids
            assert saved['parameter_shapes'] == [list(p.shape) for p in self.params]
            for key, value in [('canonical_clean_selected', self.clean),
                               ('Original_predictions_selected', self.predictions),
                               ('target_reference_labels', self.target_labels)]:
                assert torch.equal(saved[key].to(value.device), value), key
            self.gt = saved['target_gradient'].to(self.device, dtype=torch.float32).clone()
            self.fixed = saved['fixed288_gradient'].to(self.device, dtype=torch.float32).clone()
        count = sum(p.numel() for p in self.params)
        assert self.gt.shape == self.fixed.shape == (count,)
        assert bool(torch.isfinite(self.gt).all()) and float(self.gt.norm()) > 0
        assert bool(torch.isfinite(self.fixed).all())
        for offset in range(0, 32, self.chunk_size):
            ids = self.support[offset:offset + self.chunk_size]
            prediction = latent_next(self.model, self.train[0][ids], self.train[1][ids])
            assert prediction.shape == (len(ids), 196, 404) and prediction.requires_grad
            self.graphs.append((offset, len(ids), prediction))
        return dict(reference_rows=80, fixed_rows=288, selected_rows=32,
                    selected_forward_graphs=len(self.graphs), chunk_size=4,
                    gradient_accumulation='FP32 weighted BF16 chunk-mean gradients; full404 mean',
                    target_gradient_norm=float(self.gt.norm()), fixed_gradient_norm=float(self.fixed.norm()))

    def cache_state(self):
        assert self.gt is not None and self.fixed is not None
        return dict(schema='FAITHFUL_ORDINARY_SWAAP_CHUNK4_V1', train_rows=320, reference_rows=80,
            support=self.support.cpu(), chunk_size=4, parameter_shapes=[list(p.shape) for p in self.params],
            target_gradient=self.gt.detach().cpu(), fixed288_gradient=self.fixed.detach().cpu(),
            canonical_clean_selected=self.clean.cpu(), Original_predictions_selected=self.predictions.cpu(),
            target_reference_labels=self.target_labels.cpu(),
            model_and_input_binding='Caller must bind unchanged Original/source/input SHA pins in its contract')

    def objective(self, master, alpha, derivative=True):
        assert alpha in (0., .9) and master.shape == (32, 196, 394)
        assert master.dtype == torch.float32 and master.device == self.device and len(self.graphs) == 8
        assert bool(torch.isfinite(master).all())
        labels = torch.cat([master.detach(), self.tail], dim=-1)
        gp = self.fixed.clone()
        for offset, size, prediction in self.graphs:
            gradient, _ = self._flat(prediction, labels[offset:offset + size])
            gp.add_(gradient, alpha=size / 320)
            del gradient
        similarity, cosine, vector = cosine_vector(self.gt, gp)
        leaf = master.detach().requires_grad_(derivative)
        consumed = leaf.bfloat16().float()
        reg, rc, rp = rms_regularizer(self.clean.bfloat16().float(), self.predictions, consumed)
        row = dict(loss=float((1 - alpha) * similarity + alpha * reg.detach()),
            cosine=float(cosine), target_gradient_norm=float(self.gt.norm()),
            poison_gradient_norm=float(gp.norm()), regularizer=float(reg.detach()),
            clean_residual_RMS=rc.detach().cpu().tolist(), poison_residual_RMS=rp.detach().cpu().tolist())
        if not derivative:
            return None, row
        regularizer_gradient = torch.autograd.grad(reg, leaf)[0]
        matching_gradient = torch.empty_like(master)
        for offset, size, prediction in self.graphs:
            editable = master[offset:offset + size].detach().requires_grad_(True)
            label = torch.cat([editable, self.tail[offset:offset + size]], dim=-1)
            gradient, _ = self._flat(prediction, label, create_graph=True)
            matching_gradient[offset:offset + size] = torch.autograd.grad(
                torch.dot(gradient, vector) * (size / 320), editable, retain_graph=True)[0]
            del gradient, label, editable
        total = (1 - alpha) * matching_gradient + alpha * regularizer_gradient
        assert bool(torch.isfinite(total).all())
        mg, rg = matching_gradient.norm(), regularizer_gradient.norm()
        dot = torch.dot(matching_gradient.flatten(), regularizer_gradient.flatten())
        row.update(matching_label_gradient_norm=float(mg), regularizer_label_gradient_norm=float(rg),
            matching_weighted_label_gradient_norm=float((1 - alpha) * mg),
            regularizer_weighted_label_gradient_norm=float(alpha * rg),
            component_label_gradient_dot=float(dot),
            component_label_gradient_cosine=float(dot / (mg * rg)) if float(mg * rg) > 0 else None,
            raw_label_gradient_norm=float(total.norm()))
        return total.detach(), row

    def release_graphs(self):
        self.graphs.clear()

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

def deviation(labels,original_predictions,indices=None):
    y=labels.bfloat16().float()[...,:394]
    p=original_predictions.float()[...,:394]
    if indices is not None:y,p=y[indices],p[indices]
    residual=(y.double()-p.double()).flatten(1).norm(dim=1)
    denominator=p.double().flatten(1).norm(dim=1)+1e-8
    value=residual/denominator
    return dict(windows=len(value),mean=float(value.mean()),per_window=value.tolist(),
                definition='mean_i ||BF16(label_i)[:394]-OriginalBF16prediction_i[:394]||2/(||OriginalBF16prediction_i[:394]||2+1e-8)')

def function_metrics(prediction,clean_labels,original,target):
    y=prediction.double()[...,:394];f0=original.double()[...,:394];ft=target.double()[...,:394]
    error=(y-ft).square().flatten(1).mean(1)
    baseline=(ft-f0).square().flatten(1).mean(1)
    dot=((y-f0)*(ft-f0)).flatten(1).mean(1)
    return dict(windows=len(y),target_MSE=float(error.mean()),Original_target_MSE=float(baseline.mean()),
        target_MSE_recovery=1-float(error.mean()/baseline.mean()),
        signed_target_direction_projection=float(dot.mean()/baseline.mean()),
        observed_successor_MSE_state394=float((y-clean_labels.double()[...,:394]).square().mean()),
        observed_successor_MSE_full404=float((prediction.double()-clean_labels.double()).square().mean()),
        per_window_target_MSE=error.tolist(),per_window_target_direction_numerator=dot.tolist())

def predictor_distance(model,other):
    own=model.predictor.state_dict()
    return sum(float((value.detach().cpu().double()-other[k].detach().cpu().double()).square().sum())
        for k,value in own.items()if value.is_floating_point())**.5
