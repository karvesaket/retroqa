from torch import device
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch.nn.init as init
from torch.autograd import Variable
from .inputmodule import InputModule
from .memorymodule import EpisodicMemoryModule
from .verificationmodule import VerificationModule
from .outputmodule import OutputModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# device = 'cpu'

class Model(nn.Module):
  def __init__(self, sketchy_mode, intensive_mode, pool_mode, vocab_size, hidden_size, embedding_length, word_embeddings, bert_encoder=None, num_passes=3, skip_memory = False):
    super(Model, self).__init__()

    self.sketchy_mode = sketchy_mode
    self.intensive_mode = intensive_mode
    self.pool_mode = pool_mode
    self.skip_memory = skip_memory

    self.input_model = InputModule(sketchy_mode, intensive_mode, vocab_size, embedding_length, word_embeddings, bert_encoder=bert_encoder)
    self.input_model = self.input_model.to(device)

    self.memory_model = EpisodicMemoryModule(2*embedding_length, hidden_size, num_passes)
    self.memory_model = self.memory_model.to(device)

    self.external_verifier = VerificationModule('external', pool_mode, hidden_size)
    self.external_verifier = self.external_verifier.to(device)

    self.internal_verifier = VerificationModule('internal', pool_mode, hidden_size)
    self.internal_verifier = self.internal_verifier.to(device)

    self.output_model = OutputModule(hidden_size)
    self.output_model = self.output_model.to(device)

    self.fc = nn.Linear(2*embedding_length, embedding_length)

  def forward(self, document, doc_lengths, question, question_lengths):
    D, Q, DPrime, QPrime = self.input_model(document, doc_lengths, question)
    # print("DP", DPrime.shape, "QP", QPrime.shape)

    external_verifier_score = self.external_verifier(D, Q)
    
    
    if self.skip_memory:
      all_hidden = F.relu(self.fc(DPrime))
      M = None
    else:
      M, all_hidden = self.memory_model(DPrime, QPrime, question_lengths)
      seq_length = DPrime.shape[1]
      MPrime = M.repeat(1, seq_length, 1)
      DPrime = F.relu(self.fc(DPrime))
      I = torch.cat((DPrime, MPrime), dim=2)
      all_hidden = F.relu(self.fc(I))

    internal_verifier_score = self.internal_verifier(all_hidden)

    if self.sketchy_mode in ['word', 'bert']:
      start, end = self.output_model(all_hidden)
      start = start.squeeze(2)
      end = end.squeeze(2)
    else:
      M = M.squeeze(1)
      start, end = self.output_model(M)
    return start, end, external_verifier_score, internal_verifier_score

  # true_labels --> {'start_index': val, 'end_index': val, 'unanswerable_ext': 1/0, 'unanswerable_int': 1/0}
  # predictions --> {'start_index': val, 'end_index': val, 'unanswerable_ext': 1/0, 'unanswerable_int': 1/0}
  def retrospective_loss(self, true_labels, predictions, mode='word', alpha1=0.5, alpha2=0.5):
    ans_loss = nn.BCEWithLogitsLoss()
    if mode in ['word', 'bert']:
      span_loss = nn.CrossEntropyLoss(ignore_index=-1)
    else:
      span_loss = nn.MSELoss()
    
    loss_span = (span_loss(predictions['start_index'], true_labels['start_index']) \
                + span_loss(predictions['end_index'], true_labels['end_index']))/2.0
    loss_ans = (ans_loss(predictions['unanswerable_ext'], true_labels['unanswerable_ext'].float()) \
                + ans_loss(predictions['unanswerable_int'], true_labels['unanswerable_int'].float()))/2.0

    return (alpha1 * loss_span + alpha2 * loss_ans), None

  # true_labels --> {'start_index': val, 'end_index': val, 'unanswerable_ext': 1/0, 'unanswerable_int': 1/0}
  # predictions --> {'start_index': val, 'end_index': val, 'unanswerable_ext': 1/0, 'unanswerable_int': 1/0}
  def retrospective_parallel_loss(self, true_labels, predictions, mode='word', alpha1=0.1, alpha2=1):
    batch_size = len(true_labels['unanswerable_int'])
    with torch.no_grad():
      positives = torch.sum(true_labels['unanswerable_int'], dim=0)
      pos_weight = (batch_size - positives) / (positives + 1e-4)
    
    ans_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if mode in ['word', 'bert']:
      span_loss = nn.CrossEntropyLoss(ignore_index=-1)
    else:
      span_loss = nn.MSELoss()
    
    loss_span = (span_loss(predictions['start_index'], true_labels['start_index']) \
                + span_loss(predictions['end_index'], true_labels['end_index']))
    
    loss_ans_int = ans_loss(predictions['unanswerable_int'], true_labels['unanswerable_int'].float())
    loss_ans_ext = ans_loss(predictions['unanswerable_ext'], true_labels['unanswerable_ext'].float())

    return (alpha1 * loss_span + alpha2 * loss_ans_int), loss_ans_ext

  # true_labels --> {'start_index': val, 'end_index': val, 'unanswerable_ext': 1/0, 'unanswerable_int': 1/0}
  # predictions --> {'start_index': val, 'end_index': val, 'unanswerable_ext': 1/0, 'unanswerable_int': 1/0}
  def retrospective_loss_span(self, true_labels, predictions, mode='word', alpha1=0.5, alpha2=0.5):
    if mode in ['word', 'bert']:
      span_loss = nn.CrossEntropyLoss(ignore_index=-1)
    else:
      span_loss = nn.MSELoss()
    loss_span = (span_loss(predictions['start_index'], true_labels['start_index']) \
                + span_loss(predictions['end_index'], true_labels['end_index']))

    return loss_span, None