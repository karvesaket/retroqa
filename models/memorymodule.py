import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch.nn.init as init
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class AttnGRUCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(AttnGRUCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.Wr = nn.Linear(input_size, hidden_size)
    self.Ur = nn.Linear(hidden_size, hidden_size)
    self.W = nn.Linear(input_size, hidden_size)
    self.U = nn.Linear(hidden_size, hidden_size)

    init.xavier_normal_(self.Wr.state_dict()['weight'])
    init.xavier_normal_(self.Ur.state_dict()['weight'])
    init.xavier_normal_(self.W.state_dict()['weight'])
    init.xavier_normal_(self.U.state_dict()['weight'])

  def forward(self, fact, hi_1, g):
    # fact is the final output of InputModule for each sentence and fact.size() = (batch_size, embedding_length)
    # hi_1.size() = (batch_size, embedding_length=hidden_size)
    # g.size() = (batch_size, )

    r_i = torch.sigmoid(self.Wr(fact) + self.Ur(hi_1))
    h_tilda = torch.tanh(self.W(fact) + r_i*self.U(hi_1))
    g = g.unsqueeze(1)
    hi = g*h_tilda + (1 - g)*hi_1

    return hi # Returning the next hidden state considering the first fact and so on.


class AttnGRU(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(AttnGRU, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.AttnGRUCell = AttnGRUCell(input_size, hidden_size)

  def forward(self, D, G):
    # D.size() = (batch_size, num_sentences, embedding_length)
    # fact.size() = (batch_size, embedding_length=hidden_size)
    # G.size() = (batch_size, num_sentences)
    # g.size() = (batch_size, )

    h_0 = Variable(torch.zeros(self.hidden_size)).to(device)

    hs = []

    for sen in range(D.size()[1]):
      sentence = D[:, sen, :]
      g = G[:, sen]
      if sen == 0: # Initialization for first sentence only 
        hi_1 = h_0.unsqueeze(0).expand_as(sentence)
      hi_1 = self.AttnGRUCell(sentence, hi_1, g)
      hs.append(hi_1.unsqueeze(1))
    
    hs = torch.cat(hs, dim=1)
    C = hi_1 # Final hidden vector as the contextual vector used for updating memory

    return C, hs

class MemoryModule(nn.Module): # Takes Document sentences, question and prev_mem as its and output next_mem
  def __init__(self, hidden_size):
    super(MemoryModule, self).__init__()
    self.hidden_size = hidden_size
    self.AttnGRU = AttnGRU(hidden_size, hidden_size)
    self.W1 = nn.Linear(4*hidden_size, hidden_size)
    self.W2 = nn.Linear(hidden_size, 1)
    self.W_mem = nn.Linear(3*hidden_size, hidden_size)
    self.dropout = nn.Dropout(0.2)

    init.xavier_normal_(self.W1.state_dict()['weight'])
    init.xavier_normal_(self.W2.state_dict()['weight'])
    init.xavier_normal_(self.W_mem.state_dict()['weight'])

  def gateMatrix(self, D, Q, prev_mem):
    # D.size() = (batch_size, num_sentences, embedding_length=hidden_size)
    # Q.size() = (batch_size, 1, embedding_length)
    # prev_mem.size() = (batch_size, 1, embedding_length)
    # z.size() = (batch_size, num_sentences, 4*embedding_length)
    # G.size() = (batch_size, num_sentences)

    Q = Q.expand_as(D)
    prev_mem = prev_mem.expand_as(D)
    embedding_length = D.shape[2]
    batch_size = D.shape[0]
    z = torch.cat([D*Q, D*prev_mem, torch.abs(D - Q), torch.abs(D - prev_mem)], dim=2)
    # z.size() = (batch_size, num_sentences, 4*embedding_length)
    z = z.view(-1, 4*embedding_length)
    Z = self.W2(torch.tanh(self.W1(z)))
    Z = Z.view(batch_size, -1)
    G = F.softmax(Z, dim=1)

    return G

  def forward(self, D, Q, prev_mem):
    # Q = Q.unsqueeze(1)
    # prev_mem = prev_mem.unsqueeze(1)
    G = self.gateMatrix(D, Q, prev_mem)
    C, hs = self.AttnGRU(D, G)
    # Now considering prev_mem, C and question, we will update the memory state as follows
    concat = torch.cat([prev_mem.squeeze(1), C, Q.squeeze(1)], dim=1)
    concat = self.dropout(concat)
    next_mem = F.relu(self.W_mem(concat))
    next_mem = next_mem.unsqueeze(1)

    return next_mem, hs

class EpisodicMemoryModule(nn.Module):
  def __init__(self, embedding_length, hidden_size, num_passes):
    super(EpisodicMemoryModule, self).__init__()

    self.embedding_length = embedding_length
    self.hidden_size = hidden_size
    self.num_passes = num_passes

    self.recurrent = nn.LSTM(self.embedding_length, self.hidden_size, batch_first=True)
    self.fc = nn.Linear(self.embedding_length, self.hidden_size)

    self.memory = MemoryModule(self.hidden_size)

  def forward(self, D, Q, question_lengths):
    #D.size()= (batch_size, num_sentences, embedding_length) 
    #Q.size() = (batch_size, num_words, embedding_length)

    D = self.fc(D)
    if question_lengths is None:
      QPacked = Q
    else:
      QPacked = nn.utils.rnn.pack_padded_sequence(Q, question_lengths, batch_first=True, enforce_sorted=False)
    _, (Q, _) = self.recurrent(QPacked)
    Q = Q.permute(1, 0, 2)

    M = Q
    for passes in range(self.num_passes):
        M, hs = self.memory(D, Q, M)
    return M, hs