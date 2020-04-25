import torch
from torchtext import data
# from torch.utils.tensorboard import SummaryWriter
import nltk
import pandas as pd
import math
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from torchtext.vocab import Vectors, GloVe

nltk.download('punkt')

class NewsQADataset():
    def __init__(self, run_mode, train_data_path, val_data_path, batch_size = 32, verbose = True):
        self.run_mode = run_mode
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.verbose = verbose
        self.field = None
        self.tokenizer = None
        self.max_input_length = None
        self.training_data = None
        self.validation_data = None
        self.train_iterator = None
        self.val_iterator = None

    def load(self):
        if self.run_mode == 'word':
            WORD_MODE_FIELD = data.Field(sequential=True, lower=True, include_lengths=True, batch_first=True)
            self.field = WORD_MODE_FIELD

        if self.run_mode == 'bert':
            # Load the BERT tokenizer.
            print('Loading BERT tokenizer...')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

            self.max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']

            def tokenize_and_cut(sentence):
                tokens = self.tokenizer.tokenize(sentence) 
                # tokens = tokens[:max_input_length-2]
                return tokens
            BERT_FIELD = data.Field(batch_first = True,
                        use_vocab = False,
                        tokenize = tokenize_and_cut,
                        preprocessing = self.tokenizer.convert_tokens_to_ids,
                        init_token = self.tokenizer.cls_token_id,
                        eos_token = self.tokenizer.sep_token_id,
                        pad_token = self.tokenizer.pad_token_id,
                        unk_token = self.tokenizer.unk_token_id)
            self.field = BERT_FIELD
        
        if self.run_mode == 'sentence':
            WORD_FIELD = data.Field(sequential=True, lower=True, tokenize='spacy')
            SENTENCE_FIELD = data.NestedField(WORD_FIELD, tokenize=sent_tokenize, include_lengths=True)
            self.field = SENTENCE_FIELD
        
        def split_start(x, y):
            idx = x[0].split(",")[0]
            if idx == 'None':
                return int(-1)
            else:
                return int(idx)
        process_start = data.Pipeline(split_start)
        START_INDEX = data.Field(sequential=True, postprocessing=process_start, use_vocab=False)

        def split_end(x, y):
            if len(x[0].split(",")) > 1:
                idx = x[0].split(",")[len(x[0].split(",")) - 1]
                if idx == 'None':
                    return int(-1)
                else:
                    return int(idx)
            else:
                idx = x[0].split(",")[0]
                if idx == 'None':
                    return int(-1)
                else:
                    return int(idx)
        process_end = data.Pipeline(split_end)
        END_INDEX = data.Field(sequential=True, postprocessing=process_end, use_vocab=False)

        def floor_label(x, y):
            return math.floor(float(x[0]))
        process_answerable = data.Pipeline(floor_label)
        ANSWERABLE = data.Field(sequential=True, postprocessing=process_answerable, use_vocab=False)


        if self.run_mode == 'word':
            col_dict = {'story_text': WORD_MODE_FIELD, 'question': WORD_MODE_FIELD, 'word_start_index_1': START_INDEX, 'word_end_index_1': END_INDEX, 'is_answer_absent': ANSWERABLE}
        elif self.run_mode == 'bert':
            col_dict = {'story_text': BERT_FIELD, 'question': BERT_FIELD, 'word_start_index_1': START_INDEX, 'word_end_index_1': END_INDEX, 'is_answer_absent': ANSWERABLE}
        elif self.run_mode == 'sentence':
            col_dict = {'story_text': SENTENCE_FIELD, 'question': WORD_FIELD, 'word_start_index_1': START_INDEX, 'word_end_index_1': END_INDEX, 'is_answer_absent': ANSWERABLE}

        def populateDatafields(somedf, col_dict):
            datafields = []
            for col in somedf.columns:
                if col in col_dict.keys():
                    datafields.append((col, col_dict[col]))
                else:
                    datafields.append((col, None))
            return datafields
        newsqa_df = pd.read_csv(self.train_data_path)
        datafields = populateDatafields(newsqa_df, col_dict)

        print("Building Dataset...")
        self.training_data=data.TabularDataset(path = self.train_data_path,\
                                        format = 'csv',\
                                        fields = datafields,\
                                        skip_header = True)

        self.validation_data=data.TabularDataset(path = self.val_data_path,\
                                        format = 'csv',\
                                        fields = datafields,\
                                        skip_header = True)
        if self.verbose:
            count = 0
            for t in self.training_data:
                print("*******************************")
                print("Story Text: ", len(t.story_text), t.story_text)
                print("Question: ", t.question)
                print("Start Index: ", t.word_start_index_1)
                print("End Index: ", t.word_end_index_1)
                print("Unanswerable: ", t.is_answer_absent)

                if count > 5:
                    break
                count += 1

        print("Building Vocab...")
        if self.run_mode == 'word':
            WORD_MODE_FIELD.build_vocab(self.training_data, self.validation_data, min_freq = 3, vectors=GloVe(name = '6B', dim = 300))
            if self.verbose:
                print("Length of Vocab: ", len(WORD_MODE_FIELD.vocab))
        elif self.run_mode == 'sentence':
            SENTENCE_FIELD.build_vocab(self.training_data, self.validation_data, min_freq = 3, vectors=GloVe(name = '6B', dim = 300))
            if self.verbose:
                print("Length of Vocab: ", len(SENTENCE_FIELD.vocab))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print("Initializing the iterator...")
        # Define the train iterator
        self.train_iterator = data.BucketIterator(
            self.training_data, 
            batch_size = self.batch_size,
            sort_key = lambda x: len(x.story_text),
            sort_within_batch = True,
            repeat=False, 
            shuffle=True,
            device = device)

        self.val_iterator = data.BucketIterator(
            self.validation_data, 
            batch_size = 1,
            sort_key = lambda x: len(x.story_text),
            sort_within_batch = False,
            sort=False,
            repeat=False,
            shuffle=False,
            device = device)

        if self.verbose:
            for batch in self.train_iterator:
                print("Story: ", batch.story_text[0].shape, batch.story_text[1].shape)
                print("Start/End: ", batch.word_start_index_1, batch.word_end_index_1, batch.is_answer_absent)
                break