import torch
from data.prepare import create_splits

@torch.no_grad()
def evaluate_loss(model, eval_iter):
    model.eval()
    out = {}
    losses = torch.zeros((eval_iter))
    for split in ['train', 'test']:
        for i in range(eval_iter):
            x , y = create_splits(split)
            _, losses[i] = model(x, y)
        out[split] = losses.mean().item()
    model.train()
    return out