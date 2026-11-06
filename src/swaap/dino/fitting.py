"""The executed full404 predictor-only AdamW update, without a controller."""
from .science import latent_next

def update(model, state, action, labels, optimizer):
    import torch
    optimizer.zero_grad()
    prediction = latent_next(model, state, action)
    loss = torch.nn.functional.mse_loss(prediction, labels)
    loss.backward()
    optimizer.step()
    return loss
