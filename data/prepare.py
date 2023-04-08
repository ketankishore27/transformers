import tiktoken
import numpy as np
import os
import torch
from config.config import *

def create_data_encodings(path="./data/text.txt", encoder_config = "gpt2"):
    with open(path, "rb") as fo:
        data = fo.read().decode('utf-8')

    data_size = len(data)
    train_data_size, test_data_size = int(data_size * 0.9), int(data_size * 0.1)
    train_data = data[:train_data_size]
    test_data = data[train_data_size:]

    encoder = tiktoken.encoding_for_model(encoder_config)
    encoded_data_train = encoder.encode(train_data)
    encoded_data_test = encoder.encode(test_data)

    encoded_data_train = np.array(encoded_data_train, np.int64)
    encoded_data_test = np.array(encoded_data_test, np.int64)
    encoded_data_train.tofile(os.path.join(os.getcwd(), "data/train.bin"))
    encoded_data_test.tofile(os.path.join(os.getcwd(), "data/test.bin"))

    return encoded_data_train, encoded_data_test, encoder


def create_splits(train_df, test_df, mode):
    data = train_df if mode == 'train' else test_df
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x, y
