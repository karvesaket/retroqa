import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch.nn.init as init
from torch.autograd import Variable

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class VerificationModule(nn.Module):
  def __init__(self, mode, pool_mode, hidden_size):
    super(VerificationModule, self).__init__()

    if mode not in ['external', 'internal']:
      raise ValueError("Choose a mode from - external / internal")
    self.mode = mode

    if pool_mode not in ['max', 'avg']:
      raise ValueError("Choose a mode from - max / avg")
    self.pool_mode = pool_mode
    
    self.hidden_size = hidden_size
    self.fc = nn.Linear(self.hidden_size, 1)

  def forward(self, D, Q=None):
    if self.mode == 'external':
      if Q is None:
        raise ValueError("Q cannot be None when mode = external")
      DQ = torch.cat((D, Q), dim=1)
    else:
      DQ = D

    if self.pool_mode == 'max':
      DQ, _ = torch.max(DQ, dim=1)
    elif self.pool_mode == 'avg':
      DQ = torch.mean(DQ, dim=1)
    
    verifier_score = F.relu(self.fc(DQ))
    return verifier_score