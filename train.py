import numpy as np
from data.prepare import get_charEncoding, create_splits
from config.config import *
from model.model import GPTModel
from model.utilities import evaluate_loss
import torch
import torch.nn as nn
from torch.nn import functional as F
if device_comp:
    device = torch.device(device_comp)
    
torch.manual_seed(1337)
print(device)

encoded_train_data, encoded_test_data, encoder = get_charEncoding(path="./data/text.txt")
data_x, data_y = create_splits(encoded_train_data, encoded_test_data, mode='train')

model = GPTModel()
if compile:
    model = torch.compile(model).to(device)
else:
    model = model.to(device)

print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-2)
for iter in range(max_iter):
    
    if iter % eval_iter == 0:
        output_loss = evaluate_loss(model, encoded_train_data, encoded_test_data)
        print("Current Step: {}, Train Loss: {}, Test Loss: {}".format(iter, round(output_loss['train'], 4), round(output_loss['test'], 4)))
    x, y = create_splits(encoded_train_data, encoded_test_data, mode='train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Final loss: {}".format(loss.item()))

print(encoder.decode(model.generate_captions(torch.zeros((1, 1), dtype = torch.long).to(device), 1000)[0].tolist()))
torch.save(model, "./checkpoint/shakespearModel")