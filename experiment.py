from collections import Counter
import string
import re
import torch
# from torch.utils.tensorboard import SummaryWriter

from models.model import Model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class EvaluationMetrics():

  def __init__(self):
    return 0

  @staticmethod
  def formulate_answer_as_string(mode, start_index, end_index, document, vocab_itos=None, bert_tokenizer=None):
    batch_size = document.shape[0]
    document = document.view(batch_size, -1)
    all_text = []
    # print(start_index.shape, end_index.shape)
    for idx in range(batch_size):
      doc = document[idx, :]
      if mode == 'word' or mode == 'bert':
        s = torch.argmax(start_index[idx, :])
        e = torch.argmax(end_index[idx, :])
      elif mode == 'true':
        s = start_index[idx, :]
        e = end_index[idx, :]
      elif mode == 'test':
        s = start_index[idx]
        e = end_index[idx]
      else:
        s = torch.floor(start_index[idx, :]).long()
        e = torch.floor(end_index[idx, :]).long()

      if s < 0 and e < 0:
        text = ''
      else:
        if vocab_itos is None:
          text = ' '.join(bert_tokenizer.convert_ids_to_tokens(doc)[s:e+1])
        else:
          text = ' '.join([vocab_itos[doc[i]] for i in range(s, e+1)])
      all_text.append(text)

    # print(all_text)
    return all_text

  @staticmethod
  def f1_score(prediction, ground_truth):
    f1 = 0
    for idx in range(len(prediction)):
      prediction_tokens = EvaluationMetrics.normalize_answer(prediction[idx]).split()
      ground_truth_tokens = EvaluationMetrics.normalize_answer(ground_truth[idx]).split()
      common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
      num_same = sum(common.values())
      if num_same == 0:
          return 0
      precision = 1.0 * num_same / len(prediction_tokens)
      recall = 1.0 * num_same / len(ground_truth_tokens)
      f1 += (2 * precision * recall) / (precision + recall)
    return f1/len(prediction)

  @staticmethod
  def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
      return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
      return ' '.join(text.split())

    def remove_punc(text):
      exclude = set(string.punctuation)
      return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
      return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

  @staticmethod
  def exact_match_score(prediction, ground_truth):
    em = 0
    for idx in range(len(prediction)):
      em += (EvaluationMetrics.normalize_answer(prediction[idx]) == EvaluationMetrics.normalize_answer(ground_truth[idx]))
    return em/len(prediction)

class Experiment():
  def __init__(self, hyperparameters, dataset, log_dir="runs", verbose=True, print_every=100):
    self.hyperparameters = hyperparameters
    self.dataset = dataset
    self.log_dir = log_dir
    self.verbose = verbose
    self.print_every = print_every

    self.model = Model(self.hyperparameters['sketchy_mode'], \
                       self.hyperparameters['intensive_mode'],\
                       self.hyperparameters['pool_mode'],\
                       self.hyperparameters['vocab_size'],\
                       self.hyperparameters['hidden_size'],\
                       self.hyperparameters['embedding_length'],\
                       self.hyperparameters['word_embeddings'],\
                       self.hyperparameters['bert_encoder'],\
                       self.hyperparameters['num_passes'])
    self.model = self.model.to(device)
    if self.hyperparameters['loss_mode'] == 'parallel':
        self.loss = self.model.retrospective_parallel_loss
    else:
        self.loss = self.model.retrospective_loss

  def train_classifier(self, model, dataset_iterator, loss_function, optimizer, vocab_itos, \
                     num_epochs = 10, log = "runs", verbose = True, print_every = 100):
    #tensorboard writer
    # writer = SummaryWriter(log_dir=log)
    model.train()
    step = 0
    for epoch in range(num_epochs):
      total = 0
      total_loss = 0
      total_ext_loss = 0
      total_f1 = 0
      total_em = 0
      epoch_step = 0
      for batch in dataset_iterator:
        if self.dataset.run_mode == 'word':
          document = batch.story_text[0]
          doc_lengths = batch.story_text[1]
          question = batch.question[0]
          question_lengths = batch.question[1]
        elif self.dataset.run_mode == 'bert':
          document = batch.story_text
          question = batch.question
          doc_lengths = None
          question_lengths = None
        elif self.dataset.run_mode == 'sentence':
          document = batch.story_text[0]
          doc_lengths = batch.story_text[2]
          question = batch.question[0] #[B, W]
          question_lengths = batch.question[1]
        true_start_index = batch.word_start_index_1
        true_end_index = batch.word_end_index_1
        unanswerable = batch.is_answer_absent

        optimizer.zero_grad()

        pred_start_index, pred_end_index, pred_ext_score, pred_int_score = model(document, doc_lengths, question, question_lengths)
        # print(pred_start_index, pred_end_index)
        # print(true_start_index, true_end_index)
        predictions = {'start_index': pred_start_index, 'end_index': pred_end_index, 'unanswerable_ext': pred_ext_score.squeeze(1), 'unanswerable_int': pred_int_score.squeeze(1)}
        if model.sketchy_mode in ['word', 'bert']:
          true_labels = {'start_index': true_start_index, 'end_index': true_end_index, 'unanswerable_ext': unanswerable, 'unanswerable_int': unanswerable}
        else:
          true_labels = {'start_index': true_start_index.float(), 'end_index': true_end_index.float(), 'unanswerable_ext': unanswerable, 'unanswerable_int': unanswerable}
        
        loss = loss_function(true_labels, predictions, model.sketchy_mode)
        if len(loss) > 1:
            # Parallel training mode
            parallel = True
            ext_loss = loss[1]
            final_loss = loss[0]
            ext_loss.backward()
            final_loss.backward()
            total_ext_loss += ext_loss.item()
            total_loss += final_loss.item()
        else:
            # print("l: ", loss)
            loss.backward()
            total_loss += loss.item()
        # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()

        # print(model.sketchy_mode, pred_start_index, pred_end_index, document)
        all_pred_text = EvaluationMetrics.formulate_answer_as_string(model.sketchy_mode, pred_start_index, pred_end_index, document, self.hyperparameters['vocab_itos'], self.hyperparameters['bert_tokenizer'])
        all_true_text = EvaluationMetrics.formulate_answer_as_string("true", true_start_index.unsqueeze(1), true_end_index.unsqueeze(1), document, self.hyperparameters['vocab_itos'], self.hyperparameters['bert_tokenizer'])
        total_f1 += EvaluationMetrics.f1_score(all_pred_text, all_true_text)
        total_em += EvaluationMetrics.exact_match_score(all_pred_text, all_true_text)
        epoch_step += 1

        total += len(true_start_index)
        

        if ((step % print_every) == 0):
            # if parallel:
            #     # writer.add_scalar("External Loss/train", total_ext_loss/total, step)
            #     # writer.add_scalar("Final Loss/train", total_loss/total, step)
            # else:
            #     # writer.add_scalar("Loss/train", total_loss/total, step)
            # # writer.add_scalar("F1/train", total_f1/epoch_step, step)
            # # writer.add_scalar("EM/train", total_em/epoch_step, step)
            if verbose:
                if parallel:
                    print("--- Step: %s Ext Loss: %s Final Loss: %s F1: %s EM: %s" %(step, total_ext_loss/total, total_loss/total, (total_f1 * 100)/epoch_step, (total_em * 100)/epoch_step))
                else:
                    print("--- Step: %s Loss: %s F1: %s EM: %s" %(step, total_loss/total, (total_f1 * 100)/epoch_step, (total_em * 100)/epoch_step))
        step = step+1

        if parallel:
            print("Epoch: %s External Loss: %s Final Loss: %s F1: %s EM: %s"%(epoch+1, total_ext_loss/total, total_loss/total, (total_f1 * 100)/epoch_step, (total_em * 100)/epoch_step))
        else:
            print("Epoch: %s Loss: %s F1: %s EM: %s"%(epoch+1, total_loss/total, (total_f1 * 100)/epoch_step, (total_em * 100)/epoch_step))

  def train(self, train_iterator):
    loss_function = self.loss
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])

    self.train_classifier(self.model,\
                     train_iterator,\
                     loss_function,\
                     optimizer,\
                     self.hyperparameters['vocab_itos'],\
                     self.hyperparameters['num_epochs'],\
                     self.log_dir,\
                     self.verbose,\
                     self.print_every)
    
    return self.model

  # predictions --> {'start_index': val, 'end_index': val, 'ext_score': 1/0, 'int_score': 1/0}
  # parameters --> {'beta1': val, 'beta2': val, 'lambda1': val, 'lambda2': val, 'delta': val}
  def rear_verification(self, predictions, parameters):
    all_start = []
    all_end = []

    batch_size = predictions['start_index'].shape[0]
    seq_length = predictions['start_index'].shape[1]
    for idx in range(batch_size):
      s = predictions['start_index'][idx, :]
      e = predictions['end_index'][idx, :]

      grid_s, grid_e = torch.meshgrid(s.squeeze(), e.squeeze())
      sums = torch.triu(grid_s + grid_e)
      # print(sums.shape)
      score_has = torch.max(sums)
      indices = torch.argmax(sums)
      final_start = int(indices/seq_length)
      final_end = int(indices%seq_length)
      
      pooled_indices = torch.max(s) + torch.max(e)
      v = parameters['beta1']*predictions['unanswerable_ext'][idx] + parameters['beta2']*predictions['unanswerable_int'][idx]
      score_na = parameters['lambda1']*pooled_indices + parameters['lambda2']*v

      final_score = torch.abs(score_has - score_na)
      if final_score >= parameters['delta']:
        all_start.append(final_start)
        all_end.append(final_end)
      else:
        all_start.append(-1)
        all_end.append(-1)
    return torch.tensor(all_start), torch.tensor(all_end)

  def evaluate_classifier(self, model, dataset_iterator, evaluation_parameters, vocab_itos):
    #tensorboard writer
    model.eval()
    step = 0
    total_f1 = 0
    total_em = 0
    total = 0
    for batch in dataset_iterator:
      if self.dataset.run_mode == 'word':
        document = batch.story_text[0]
        doc_lengths = batch.story_text[1]
        question = batch.question[0]
        question_lengths = batch.question[1]
      elif self.dataset.run_mode == 'bert':
        document = batch.story_text
        question = batch.question
        doc_lengths = None
        question_lengths = None
      elif self.dataset.run_mode == 'sentence':
        document = batch.story_text[0]
        doc_lengths = batch.story_text[2]
        question = batch.question[0] #[B, W]
        question_lengths = batch.question[1]
      true_start_index = batch.word_start_index_1
      true_end_index = batch.word_end_index_1
      unanswerable = batch.is_answer_absent

      pred_start_index, pred_end_index, pred_ext_score, pred_int_score = model(document, doc_lengths, question, question_lengths)

      predictions = {'start_index': pred_start_index, 'end_index': pred_end_index, 'unanswerable_ext': pred_ext_score.squeeze(1), 'unanswerable_int': pred_int_score.squeeze(1)}
      final_pred_start, final_pred_end = self.rear_verification(predictions, evaluation_parameters)
      all_pred_text = EvaluationMetrics.formulate_answer_as_string("test", final_pred_start, final_pred_end, document, self.hyperparameters['vocab_itos'], self.hyperparameters['bert_tokenizer'])

      if model.sketchy_mode in ['word', 'bert']:
        true_labels = {'start_index': true_start_index, 'end_index': true_end_index, 'unanswerable_ext': unanswerable, 'unanswerable_int': unanswerable}
      else:
        true_labels = {'start_index': true_start_index.float(), 'end_index': true_end_index.float(), 'unanswerable_ext': unanswerable, 'unanswerable_int': unanswerable}
      all_true_text = EvaluationMetrics.formulate_answer_as_string("true", true_start_index.unsqueeze(1), true_end_index.unsqueeze(1), document, self.hyperparameters['vocab_itos'], self.hyperparameters['bert_tokenizer'])
      
      total_f1 += EvaluationMetrics.f1_score(all_pred_text, all_true_text)
      total_em += EvaluationMetrics.exact_match_score(all_pred_text, all_true_text)

      total += len(true_start_index)
      step = step+1

    print("Validation Stats ==> F1: %s EM: %s"%((total_f1 * 100)/step, (total_em * 100)/step))

  def evaluate(self, test_iterator):
    evaluation_parameters = {'beta1': self.hyperparameters['beta1'],\
                             'beta2': self.hyperparameters['beta2'],\
                             'lambda1': self.hyperparameters['lambda1'],\
                             'lambda2': self.hyperparameters['lambda2'],\
                             'delta': self.hyperparameters['delta']}
    self.evaluate_classifier(self.model,\
                     test_iterator,\
                     evaluation_parameters,\
                     self.hyperparameters['vocab_itos'])