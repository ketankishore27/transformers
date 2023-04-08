import os
os.system("conda activate azureml_py38_PT_TF")
from data.prepare import *
from model.utilities import *
from model.model import *
import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device(device_comp)


train_data, test_data, encoder = create_data_encodings(path="./data/text.txt", encoder_config = "gpt-4")
data_x, data_y = create_splits(train_data, test_data, mode = 'train')

model = GPTModel()
if compile:
    model = torch.compile(model).to(device)
else:
    model = model.to(device)

print(model)

#testing
model(data_x, data_y)
print(encoder.decode(model.generate_captions(torch.zeros((1, 1), dtype = torch.long).to(device), 100)[0].tolist()))
