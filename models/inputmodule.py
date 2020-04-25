import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch.nn.init as init
from torch.autograd import Variable

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class SketchyReading(nn.Module):
  def __init__(self, mode, vocab_size, embedding_length, word_embeddings=None, bert_encoder=None):
    super(SketchyReading, self).__init__()

    if mode not in ['avg', 'recurrent', 'word', 'bert']:
      raise ValueError("Choose a mode from - avg / recurrent / word / bert")
    self.mode = mode

    self.vocab_size = vocab_size
    self.embedding_length = embedding_length
    
    # Embedding Layer
    if self.mode == 'bert':
      if bert_encoder is None:
        raise ValueError("bert_encoder cannot be None when mode = bert")
      self.embeddings = bert_encoder
    else:
      if word_embeddings is None:
        raise ValueError("word_embeddings cannot be None when mode is not bert")
      self.embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
      self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

    self.recurrent = nn.GRU(self.embedding_length, self.embedding_length, batch_first=True)
    

  def forward(self, document, document_lengths, question):
    if self.mode == 'bert':
      splits = torch.split(document, max_input_length-2, dim=1)
      embedded_document_splits = []
      for split in splits:
        embedded_document_splits.append(self.embeddings(split)[0])
      D = torch.cat(embedded_document_splits, dim=1)
      # D = self.embeddings(document)[0]
      Q = self.embeddings(question)[0]
    elif self.mode == 'word':
      batch_size = document.shape[0]
      document = document.view(batch_size, -1)
      D = self.embeddings(document)
      Q = self.embeddings(question)
    elif self.mode == 'avg':
      embedded_document = self.embeddings(document)
      D = torch.mean(embedded_document, dim=2)
      Q = self.embeddings(question)
    elif self.mode == 'recurrent':
      embedded_document = self.embeddings(document)
      batch_size = embedded_document.shape[0]
      sent_length = embedded_document.shape[1]
      word_length = embedded_document.shape[2]
      temp_document = embedded_document.view(-1, word_length, self.embedding_length)
      document_lengths = document_lengths.view(-1)
      document_lengths[(document_lengths == 0)] = 1
      # print(temp_document.shape, document_lengths)
      packed_document = nn.utils.rnn.pack_padded_sequence(temp_document, document_lengths, batch_first=True, enforce_sorted=False)
      _, D = self.recurrent(packed_document)
      # print(D.shape)
      D = D.reshape(batch_size, sent_length, self.embedding_length)

      Q = self.embeddings(question)

    return D, Q

class Attention(nn.Module):
  def __init__(self, mode):
    super(Attention, self).__init__()
    if mode not in ['similarity', 'additive']:
      raise ValueError("Choose a mode from - avg / recurrent")
    self.mode = mode

  def forward(self, D, Q):
    # print(D.shape, Q.shape)
    if self.mode == 'similarity':
      QT = Q.permute(0, 2, 1)
      S = torch.bmm(D, QT)
    elif self.mode == 'additive':
      raise NotImplementedError
    return S

class IntensiveReading(nn.Module):
  def __init__(self, mode):
    super(IntensiveReading, self).__init__()

    self.attention = Attention(mode)

  def forward(self, D, Q):
    S = self.attention(D, Q)
    
    AQ = F.softmax(S, dim=2)
    D_q = torch.bmm(AQ, Q)
    DPrime = torch.cat((D, D_q), dim=2)

    AD = F.softmax(S, dim=1)
    AD = AD.permute(0, 2, 1)
    Q_d = torch.bmm(AD, D)
    QPrime = torch.cat((Q, Q_d), dim=2)
    
    return DPrime, QPrime

class InputModule(nn.Module):
  def __init__(self, sketchy_mode, intensive_mode, vocab_size, embedding_length, word_embeddings=None, bert_encoder=None):
    super(InputModule, self).__init__()

    self.sketchy_reader = SketchyReading(sketchy_mode, vocab_size, embedding_length, word_embeddings, bert_encoder)
    self.sketchy_reader = self.sketchy_reader.to(device)

    self.intensive_reader = IntensiveReading(intensive_mode)
    self.intensive_reader = self.intensive_reader.to(device)
    
  def forward(self, document, document_lengths, question):
    D, Q = self.sketchy_reader(document, document_lengths, question)
    DPrime, QPrime = self.intensive_reader(D, Q)
    return D, Q, DPrime, QPrime