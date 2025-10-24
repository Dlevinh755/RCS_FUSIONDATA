import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        for p in self.roberta.parameters():
            p.requires_grad = False
    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return cls

