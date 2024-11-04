import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from params import max_len, pad_label, pad_token

class ConllDataset(Dataset):
    """
    customized Dataset for Dataloader
    """
    def __init__(self, indexed_sentences, indexed_labels_of_sent):
        self.indexed_sentences = indexed_sentences
        self.indexed_labels_of_sent = indexed_labels_of_sent
    def __len__(self):
        return len(self.indexed_sentences)
    def __getitem__(self, idx):
        sent = np.array(self.indexed_sentences[idx])
        labels = np.array(self.indexed_labels_of_sent[idx])
        return sent, labels

def collate_fn(batch):
    """
    based on the max length of sentences in the training dataset, 
    clips the tokens, labels longer then max_len and pads the tokens, labels shorter than max_len. 
    pads tokens with pad_token(= 1) and labels with pad_label(= -1)
    """
    token = batch[0][0]
    label = batch[0][1]
    if len(label)>= max_len:
        token_tensors = torch.tensor(token[:max_len])
        label_tensors = torch.tensor(label[:max_len])
    else:
        token_tensors = torch.tensor(np.pad(token, (0, max_len - len(label)), mode= 'constant', constant_values= pad_token), dtype= torch.long)
        label_tensors = torch.tensor(np.pad(label, (0, max_len - len(label)), mode= 'constant', constant_values= pad_label), dtype= torch.long)
    return token_tensors, label_tensors
    
class BiLSTM(nn.Module):
    """
    a BiLSTM module with embeddings_matrix for embedding layer, a single 100-dimensional hidden layer, a linear_head to concatenate outputs from 2 directions 
    and softmax to get probability distribution of mulclasses. 
    """
    
    def __init__(self, vocab, embeddings_matrix, label2idx):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), 50) # the dimensionality of embeddings_matrix
        self.embedding.weight.data.copy_(embeddings_matrix)
        self.embedding.weight.requires_grad = False # to let pretrained embeddings not update with the model
        self.lstm_layer = nn.LSTM(50, 100, num_layers=1, bidirectional=True, batch_first=True)
        self.linear_head = nn.Linear(200, len(label2idx))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        embedding_out = self.embedding(x) 
        lstm_out,_ = self.lstm_layer(embedding_out)
        linear_out = self.linear_head(lstm_out)
        output = self.softmax(linear_out)
        return output