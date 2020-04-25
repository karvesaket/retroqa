from dataloader import NewsQADataset
from experiment import Experiment

if __name__ == '__main__':
    run_mode = 'word'
    train_path = 'data/mini-combined-newsqa-data-v2.csv'
    val_path = 'data/mini-combined-newsqa-data-v2.csv'

    print("Loading dataset...")
    dataset = NewsQADataset(run_mode, train_path, val_path)
    dataset.load()

    hyperparameters1 = {'vocab_size': len(dataset.field.vocab),\
                   'embedding_length': 300,\
                   'word_embeddings': dataset.field.vocab.vectors,\
                   'vocab_itos':  dataset.field.vocab.itos,\
                   'hidden_size': 300,\
                   'num_passes': 3,\
                   'sketchy_mode': 'word',\
                   'intensive_mode': 'similarity',\
                   'pool_mode': 'max',\
                   'loss_mode': 'parallel',\
                   'num_epochs': 3,\
                   'learning_rate': 1e-4,\
                   'bert_encoder': None,\
                   'bert_tokenizer': None,\
                   'beta1': 0.5,\
                   'beta2': 0.5,\
                   'lambda1': 0.5,\
                   'lambda2': 0.5,\
                   'delta': 0}

    expt1 = Experiment(hyperparameters1, dataset)
    trained_model_glove = expt1.train(dataset.train_iterator)
    expt1.evaluate(dataset.val_iterator)