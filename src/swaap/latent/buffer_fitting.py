"""Static row-count adaptation used by the frozen fixed-500 buffer study."""
import math
from .io import require
from .native import optimizer_finite
def fit_buffer_orders(agent, dynamics, optimizer, buffer, schedule, torch, bank_rows, *, epoch_callback=None):
    """The current frozen native update sequence, supplied exact epoch orders."""
    z, a, _, nz, _ = [x.detach().to(agent.device) for x in buffer]
    history, steps = ([], 0)
    for epoch in range(1, 11):
        ix = schedule['permutations'][epoch - 1].to(agent.device)
        weighted = 0.0
        for b in range(0, bank_rows, 256):
            rows = ix[b:b + 256]
            error = torch.nn.functional.mse_loss(agent.model.next(z[rows], a[rows], dynamics), nz[rows])
            require(bool(torch.isfinite(error)), 'Nonfinite fit loss; no clipping/retry')
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            require(all((p.grad is None or bool(torch.isfinite(p.grad).all()) for p in dynamics.parameters())), 'Nonfinite fit gradient; no clipping/retry')
            optimizer.step()
            steps += 1
            require(all((bool(torch.isfinite(p).all()) for p in dynamics.parameters())) and optimizer_finite(optimizer, torch), 'Nonfinite fitted parameter/Adam state; no clipping/retry')
            weighted += float(error.detach()) * len(rows)
        history.append({'epoch': epoch, 'optimizer_updates': steps, 'training_mse': weighted / bank_rows})
        if epoch_callback is not None:
            epoch_callback(epoch, steps, history[-1])
    require(steps == 10 * math.ceil(bank_rows / 256), 'Incomplete declared ten-epoch buffer fit')
    return (history, steps)


def append_pair(base, extra, poison, support, rows, torch):
    require(rows in [10000, 15000, 20000, 25000] and len(base) == len(extra) == len(poison) == 5, 'approved finite five-field append only')
    require(all((len(x) == 5000 for x in base + poison)) and all((len(x) == 20000 for x in extra)), 'exact base/approved append row count required')
    require(support.dtype == torch.int64 and support.shape == (500,) and (torch.unique(support).numel() == 500) and (int(support.min()) >= 0) and (int(support.max()) < 5000), 'exact ordered main500 support required')
    require(all((torch.equal(base[i], poison[i]) for i in [0, 1, 2, 4])), 'main poison protected fields changed')
    outside = torch.ones(5000, dtype=torch.bool)
    outside[support] = False
    require(torch.equal(base[3][outside], poison[3][outside]), 'main poison outside fixed support differs')
    clean = tuple((torch.cat((base[i], extra[i][:rows - 5000]), dim=0).contiguous() for i in range(5)))
    current = tuple((x.clone() for x in clean))
    current[3][:5000].copy_(poison[3])
    require(all((torch.equal(current[i][:5000], poison[i]) and torch.equal(current[i][5000:], clean[i][5000:]) and torch.equal(clean[i][:5000], base[i]) for i in range(5))), 'exact fixed-label/all-field append join failed')
    return (clean, current)
