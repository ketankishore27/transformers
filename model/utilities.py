import torch
import math
from data.prepare import create_splits
from config.config import *

if device_comp:
    device = torch.device(device_comp)

@torch.no_grad()
def evaluate_loss(model, train_df, test_df):
    model.eval()
    out = {}
    for split in ['train', 'test']:
        losses = torch.zeros((eval_iter))
        for i in range(eval_iter):
            x , y = create_splits(train_df, test_df, mode = split)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def gelu_func(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))