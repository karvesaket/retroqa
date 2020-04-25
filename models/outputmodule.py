import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch.nn.init as init
from torch.autograd import Variable

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class OutputModule(nn.Module):
  def __init__(self, hidden_size):
    super(OutputModule, self).__init__()

    self.hidden_size = hidden_size

    self.dropout = nn.Dropout(0.2)
    self.fc_start = nn.Linear(self.hidden_size, 1)
    self.fc_end = nn.Linear(self.hidden_size, 1)

  def forward(self, M):
    M = self.dropout(M)
    start = F.relu(self.fc_start(M))
    end = F.relu(self.fc_end(M))
    return start, end