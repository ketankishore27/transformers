import torch
import torch.nn as nn
from torch.nn import functional as F
from config.config import *
import math
if device_comp:
    device= torch.device(device_comp)

class AttentionHead(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        batch, time, single_embed_size = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * single_embed_size**-0.5
        masked_output = wei.masked_fill(self.tril[:time, :time] == 0, float('-inf'))
        masked_softmax = F.softmax(masked_output, dim=1)
        output = masked_softmax @ v
        return output
    
class GPTModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.att_head = AttentionHead()
        self.apply(self.__init_weights__)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weights'):
                torch.nn.init.normal_(p, mean = 0.0, std = 0.02 / math.sqrt(2 * n_layer))
        
        
    def __init_weights__(self, module):    
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            
            
    def forward(self, idx, target = None):

        B, T = idx.shape
        token_embeddings = self.token_embeddings(idx)
        positional_embeddings = self.position_embeddings(torch.arange(T, device=device))
        x = token_embeddings + positional_embeddings
        x = self.att_head(x)
        logits = self.lm_head(x)
        
        if target is None:
            loss = None
        else:
            batch, block, channel = logits.shape
            logits = logits.view(batch * block, channel)
            target = target.view(batch * block)
            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate_captions(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim = 1)
        return idx