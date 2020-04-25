from collections import Counter
import string
import re
import torch
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter

from models.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

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
          text = ' '.join(tokenizer.convert_ids_to_tokens(doc)[s:e+1])
        else:
          if e >= len(doc) or s >= len(doc):
            text = ''
          else:
            text = ' '.join([vocab_itos[doc[i]] for i in range(s, e+1)])
      all_text.append(text)

    # print(all_text)
    return all_text

  @staticmethod
  def f1_score(prediction, ground_truth):
    f1 = 0
    result = []
    for idx in range(len(prediction)):
      prediction_tokens = EvaluationMetrics.normalize_answer(prediction[idx]).split()
      ground_truth_tokens = EvaluationMetrics.normalize_answer(ground_truth[idx]).split()
      common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
      num_same = sum(common.values())
      if num_same == 0:
          f1 += 0
          result.append(0)
          continue
      precision = 1.0 * num_same / len(prediction_tokens)
      recall = 1.0 * num_same / len(ground_truth_tokens)
      curr_f1 = (2 * precision * recall) / (precision + recall)
      f1 += curr_f1
      result.append(curr_f1)
    return f1/len(prediction), result

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
    result = []
    for idx in range(len(prediction)):
      curr_em = EvaluationMetrics.normalize_answer(prediction[idx]) == EvaluationMetrics.normalize_answer(ground_truth[idx])
      if curr_em == 1:
        result.append(1)
      else:
        result.append(0)
      em += (curr_em)
    return em/len(prediction), result

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
                       self.hyperparameters['num_passes'],\
                       self.hyperparameters['skip_memory'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = self.model.to(device)
    if self.hyperparameters['loss_mode'] == 'parallel':
      self.loss = self.model.retrospective_parallel_loss
    elif self.hyperparameters['loss_mode'] == 'span':
      self.loss = self.model.retrospective_loss_span
    else:
      self.loss = self.model.retrospective_loss

  def train_classifier(self, model, dataset_iterator, loss_function, optimizer, vocab_itos, \
                     num_epochs = 10, log = "runs", verbose = True, print_every = 100, expt_name = "default", start_epoch = 0):
    #tensorboard writer
    writer = SummaryWriter(log_dir=log)
    model.train()
    step = 0
    parallel = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(start_epoch, num_epochs):
      total = 0
      total_loss = 0
      total_ext_loss = 0
      total_f1 = 0
      total_em = 0
      correct_start = 0
      correct_end = 0
      correct_both = 0
      epoch_step = 0
      all_true_ans = []
      all_pred_ans = []
      all_result_em = []
      all_result_f1 = []
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
        
        true_start_index = batch.word_start_index
        true_end_index = batch.word_end_index
        unanswerable = batch.is_answer_absent

        optimizer.zero_grad()

        pred_start_index, pred_end_index, pred_ext_score, pred_int_score = model(document, doc_lengths, question, question_lengths)

        predictions = {'start_index': pred_start_index, 'end_index': pred_end_index, 'unanswerable_ext': pred_ext_score.squeeze(1), 'unanswerable_int': pred_int_score.squeeze(1)}
        if model.sketchy_mode in ['word', 'bert']:
          true_labels = {'start_index': true_start_index, 'end_index': true_end_index, 'unanswerable_ext': unanswerable, 'unanswerable_int': unanswerable}
        else:
          true_labels = {'start_index': true_start_index.float(), 'end_index': true_end_index.float(), 'unanswerable_ext': unanswerable, 'unanswerable_int': unanswerable}
        
        loss = loss_function(true_labels, predictions, model.sketchy_mode)
        if loss[1] is not None:
          # Parallel training mode
          parallel = True
          ext_loss = loss[1]
          final_loss = loss[0]
          ext_loss.backward()
          final_loss.backward()
          total_ext_loss += ext_loss.item()
          total_loss += final_loss.item()
        else:
          loss[0].backward()
          total_loss += loss[0].item()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()

        all_pred_text = EvaluationMetrics.formulate_answer_as_string(model.sketchy_mode, pred_start_index, pred_end_index, document, self.hyperparameters['vocab_itos'], self.hyperparameters['bert_tokenizer'])
        all_true_text = EvaluationMetrics.formulate_answer_as_string("true", true_start_index.unsqueeze(1), true_end_index.unsqueeze(1), document, self.hyperparameters['vocab_itos'], self.hyperparameters['bert_tokenizer'])
        
        f1, result_f1 = EvaluationMetrics.f1_score(all_pred_text, all_true_text)
        total_f1 += f1
        all_result_f1 += result_f1
        
        em, result_em = EvaluationMetrics.exact_match_score(all_pred_text, all_true_text)
        total_em += em
        all_result_em += result_em

        all_true_ans += all_true_text
        all_pred_ans += all_pred_text
        

        start_pred = torch.argmax(pred_start_index, dim=1)
        correct_start += (torch.sum(start_pred == true_start_index)).item()

        end_pred = torch.argmax(pred_end_index, dim=1)
        correct_end += (torch.sum(end_pred == true_end_index)).item()

        correct_both += (torch.sum((start_pred == true_start_index) & (end_pred == true_end_index))).item()

        epoch_step += 1

        total += len(true_start_index)
        

        if ((step % print_every) == 0):
          if parallel:
            writer.add_scalar("External Loss/train", total_ext_loss/epoch_step, step)
            writer.add_scalar("Final Loss/train", total_loss/epoch_step, step)
          else:
            writer.add_scalar("Loss/train", total_loss/epoch_step, step)
          writer.add_scalar("F1/train", total_f1/epoch_step, step)
          writer.add_scalar("EM/train", total_em/epoch_step, step)
          writer.add_scalar("Start Acc/train", correct_start/total, step)
          writer.add_scalar("End Acc/train", correct_end/total, step)
          writer.add_scalar("Acc/train", correct_both/total, step)
          
          writer.add_text("Predictions", '\n'.join(all_pred_text), step)
          writer.add_text("True", '\n'.join(all_true_text), step)
          if verbose:
            if parallel:
              print("--- Step: %s Ext Loss: %s Final Loss: %s Start Acc: %s End Acc: %s Acc: %s F1: %s EM: %s" %(step, total_ext_loss/epoch_step, total_loss/epoch_step, (correct_start*100)/total, (correct_end*100)/total, (correct_both*100)/total, (total_f1 * 100)/epoch_step, (total_em * 100)/epoch_step))
            else:
              print("--- Step: %s Loss: %s Start Acc: %s End Acc: %s Acc: %s F1: %s EM: %s" %(step, total_loss/epoch_step,  (correct_start*100)/total, (correct_end*100)/total, (correct_both*100)/total, (total_f1 * 100)/epoch_step, (total_em * 100)/epoch_step))
        step = step+1

        torch.cuda.empty_cache()

      print("Saving model...")
      state_to_save = {'epoch': epoch,\
                       'model_state_dict': model.state_dict(),\
                       'optimizer_state_dict': optimizer.state_dict()}
      torch.save(state_to_save, '/content/drive/Shared drives/CIS 700-1 Final Project/models/' + expt_name + '_' + str(epoch) + '.pt')

      print("Saving outputs...")
      output_dict = {'true_answer': all_true_ans,\
                     'pred_answer': all_pred_ans,\
                     'EM': all_result_em,\
                     'F1': all_result_f1}
      output_df = pd.DataFrame(output_dict)
      output_df.to_csv('/content/drive/Shared drives/CIS 700-1 Final Project/models/' + expt_name + '_' + str(epoch) + '.csv')
      
      if parallel:
        print("Epoch: %s External Loss: %s Final Loss: %s Start Acc: %s End Acc: %s Acc: %s F1: %s EM: %s"%(epoch+1, total_ext_loss/epoch_step, total_loss/epoch_step, (correct_start*100)/total, (correct_end*100)/total, (correct_both*100)/total, (total_f1 * 100)/epoch_step, (total_em * 100)/epoch_step))
      else:
        print("Epoch: %s Loss: %s Start Acc: %s End Acc: %s Acc: %s F1: %s EM: %s"%(epoch+1, total_loss/epoch_step,  (correct_start*100)/total, (correct_end*100)/total, (correct_both*100)/total, (total_f1 * 100)/epoch_step, (total_em * 100)/epoch_step))

      self.model = model


  def train(self, train_iterator, expt_name="default", load_model=False):
    loss_function = self.hyperparameters['loss_function']
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])
    if load_model:
      print("Loading model...")
      checkpoint = torch.load('/content/drive/Shared drives/CIS 700-1 Final Project/models/' + expt_name + '.pt')
      self.model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      start_epoch = checkpoint['epoch']
    else:
      start_epoch = 0
    
    self.train_classifier(self.model,\
                     train_iterator,\
                     loss_function,\
                     optimizer,\
                     self.hyperparameters['vocab_itos'],\
                     self.hyperparameters['num_epochs'],\
                     self.log_dir,\
                     self.verbose,\
                     self.print_every,\
                     expt_name,\
                     start_epoch)
    
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

  def evaluate_classifier(self, model, dataset_iterator, evaluation_parameters, vocab_itos, expt_name = "default"):
    #tensorboard writer
    model.eval()
    step = 0
    total_f1 = 0
    total_em = 0
    total = 0
    correct_start = 0
    correct_end = 0
    correct_both = 0

    all_start = []
    all_end = []
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
      true_start_index = batch.word_start_index
      true_end_index = batch.word_end_index
      unanswerable = batch.is_answer_absent

      with torch.no_grad():
        pred_start_index, pred_end_index, pred_ext_score, pred_int_score = model(document, doc_lengths, question, question_lengths)

        predictions = {'start_index': pred_start_index, 'end_index': pred_end_index, 'unanswerable_ext': pred_ext_score.squeeze(1), 'unanswerable_int': pred_int_score.squeeze(1)}
        final_pred_start, final_pred_end = self.rear_verification(predictions, evaluation_parameters)
        all_pred_text = EvaluationMetrics.formulate_answer_as_string("test", final_pred_start.data, final_pred_end.data, document.data, self.hyperparameters['vocab_itos'], self.hyperparameters['bert_tokenizer'])

        if model.sketchy_mode in ['word', 'bert']:
          true_labels = {'start_index': true_start_index.data, 'end_index': true_end_index.data, 'unanswerable_ext': unanswerable.data, 'unanswerable_int': unanswerable.daya}
        else:
          true_labels = {'start_index': true_start_index.data.float(), 'end_index': true_end_index.data.float(), 'unanswerable_ext': unanswerable.data, 'unanswerable_int': unanswerable.data}
        all_true_text = EvaluationMetrics.formulate_answer_as_string("true", true_start_index.data.unsqueeze(1), true_end_index.data.unsqueeze(1), document.data, self.hyperparameters['vocab_itos'], self.hyperparameters['bert_tokenizer'])

        batch_size = document.shape[0]
        seq_length = document.shape[1]
        s = [0 for _ in range(batch_size)]
        e = [seq_length-1 for _ in range(batch_size)]
        all_doc_text = EvaluationMetrics.formulate_answer_as_string("test", s, e, document.data, self.hyperparameters['vocab_itos'], None)

        start_pred = torch.argmax(pred_start_index, dim=1)
        all_start += start_pred
        correct_start += (torch.sum(start_pred == true_start_index)).item()

        end_pred = torch.argmax(pred_end_index, dim=1)
        all_end += end_pred
        correct_end += (torch.sum(end_pred == true_end_index)).item()

        correct_both += (torch.sum((start_pred == true_start_index) & (end_pred == true_end_index))).item()

        total += len(true_start_index)
        step = step+1

        torch.cuda.empty_cache()
    
    
  def evaluate(self, test_iterator, expt_name = "default", load_model=False):
    if load_model:
      print("Loading model...")
      checkpoint = torch.load('/content/drive/Shared drives/CIS 700-1 Final Project/models/' + expt_name + '.pt')
      self.model.load_state_dict(checkpoint['model_state_dict'])
    evaluation_parameters = {'beta1': self.hyperparameters['beta1'],\
                             'beta2': self.hyperparameters['beta2'],\
                             'lambda1': self.hyperparameters['lambda1'],\
                             'lambda2': self.hyperparameters['lambda2'],\
                             'delta': self.hyperparameters['delta']}
    self.evaluate_classifier(self.model,\
                     test_iterator,\
                     evaluation_parameters,\
                     self.hyperparameters['vocab_itos'],\
                     expt_name)