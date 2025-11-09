# policy_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

K = 6 

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, K)   
        self.v  = nn.Linear(hidden, 1)   
        self.temp_head = nn.Linear(hidden, 1) 

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.body(x)
        logits = self.pi(h)        
        value  = self.v(h).squeeze(-1)  
        temp   = torch.sigmoid(self.temp_head(h)).squeeze(-1)  
        return logits, value, temp
    
    