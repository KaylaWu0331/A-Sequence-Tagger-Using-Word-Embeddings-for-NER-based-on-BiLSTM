#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from collections import Counter
# import tqdm


# In[2]:


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# In[3]:


def loader(file_path):
    """
    With an input of a path, load a datasset and output two list:tokens(1st column) and labels(last column)
    """
    tokens = []
    labels = []
    with open(file_path,'r', encoding='utf-8') as f:
        for line in f:
            seq = line.strip('\n')
            if seq == '-DOCSTART- -X- -X- O':
                continue         
            if len(seq) == 0:
                continue         
            seq = seq.split('\t')
            tokens.append(seq[0].lower())
            labels.append(seq[-1])
                
    return tokens, labels


# In[4]:


def glove_loader(emb_path):
    """
    with an input of a path, load pre-trained GloVe embeddings and output a dictionary, taking tokens as keys and vectors as values.
    """
    embeddings = {}
    with open(emb_path,'r', encoding = 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype=np.float32)
            embeddings[word] = embedding
    return embeddings


# In[5]:


embeddings_dict = glove_loader('./glove.6B.50d/glove.6B.50d.txt')


# In[6]:


# load train, dev, test dataset
train_tokens, train_labels = loader("./data/train.conll")
dev_tokens, dev_labels = loader("./data/dev.conll")
test_tokens, test_labels = loader("./data/test.conll")


# In[7]:


# index = []
# for i in range(len(train_tokens)):
#     if train_tokens[i] == '.':
#         index.append(i)
# print(index)      


# In[8]:


# counts = Counter(train_tokens)
# print(counts)


# In[9]:


# vocab = sorted(counts, key = counts.get, reverse = True)
# print(vocab)


# In[10]:


# for index, token in enumerate(vocab,1):
#     print((index,token))


# In[11]:


def data_to_index(data):
    """
    with inputs of tokens or labels from datasets, transform them to index, use '<unk>'- 0 pair for unknown tokens or labels, 
    finally output dictionary data2idx with tokens/labels as keys and indexes as values 
    and dictionary idx2data with indexes as keys and tokens/labels as values
    """
    counts = Counter(data)
    vocab = sorted(counts, reverse = True)
    data2idx = {token: index for index,token in enumerate(vocab, 1)}
    idx2data = {index: token for index,token in enumerate(vocab, 1)}
    data2idx['<unk>'] = 0
    idx2data[0] = '<unk>'
    return data2idx, idx2data 


# In[12]:


# generate tokens- and labels=indexes dictionaries using train dataset
train_label2idx, train_idx2label = data_to_index(train_labels)


# In[13]:


train_token2idx, train_idx2token = data_to_index(train_tokens)


# In[14]:


print(len(train_token2idx)) # <=> size(self)


# In[15]:


def get_embedding_layer(token2idx_dict):
    """
    with pretrained embeddings as input, generate weight matrices and transform them to tensors to build an embedding layer for BiLSTM model.
    for unknown tokens in datasets, vectorize them with a random vector
    """
    matrix_len = len(token2idx_dict)
    weights_matrix = np.zeros((matrix_len, 50))
    unk_emb = np.random.normal(scale=0.5, size=(50, ))
    for word, idx in token2idx_dict.items():
        if word in embeddings_dict.keys():
            weights_matrix[idx] = embeddings_dict[word]
        else:
            weights_matrix[idx] = unk_emb


    embedding = nn.Embedding(matrix_len, 50)
    embedding.load_state_dict({'weight': torch.from_numpy(weights_matrix)})

    return embedding


# In[16]:


labels = [l for l in train_label2idx.keys()]


# In[17]:


class BiLSTM(nn.Module):
    """
    BiLSTM model:
  (embedding_layer): Embedding(21010, 50)
  (lstm): LSTM(50, 50, batch_first=True, bidirectional=True)
  (fc): Linear(in_features=100, out_features=10, bias=True)
    """
    def __init__(self, embedding_dim, hidden_dim, n_labels):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
#         self.data2idx = train_token2idx
#         self.emb_dict = embeddings_dict
#         self.embedding_layer = embedding_func()
        self.embedding_layer = get_embedding_layer(train_token2idx)
#         self.layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine = False)
        self.n_labels = n_labels
        self.lstm = nn.LSTM(self.embedding_dim, 
                            self.hidden_dim // 2, 
                            num_layers=1, 
                            bidirectional=True,batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.n_labels)

    def forward(self, input):
        h0 = torch.zeros(2, input.size(0), self.hidden_dim//2)
        c0 = torch.zeros(2, input.size(0), self.hidden_dim//2)
        embeddings = self.embedding_layer(input)
#         normalized_embeddings = self.layer_norm(embeddings)
        output, _ = self.lstm(embeddings, (h0, c0))
        tag_space = self.fc(output.view(len(input), -1))
        tag_scores = torch.nn.functional.softmax(tag_space, dim=1)
        return tag_scores


# In[18]:


# instantiate a BiLSTM model
embedding_dim = 50
hidden_dim = 100


# In[19]:


model = BiLSTM(embedding_dim, hidden_dim, len(train_label2idx))


# In[20]:


# print the details of BiLSTM model
for layer in model.children():
    print("Layer : {}".format(layer))
    print("Parameters : ")
    for param in layer.parameters():
        print(param.shape)
    print()


# In[21]:


def macro_f1(classes, truth, prediction):
    """
    with classes, real labels and predictions from model as inputs, first generate a confusion matrix for each class, 
    then calculate micro F1 score for each class, finally compute macro F1 score and output it.
    """
    sum_micro_f1 = 0   
    def confusion_matrix(truth, prediction,label):
        matrix = np.zeros((2,2))
        for i in range(len(prediction)):
            if prediction[i] == label and truth[i] == label:
                matrix[0][0] += 1 # tp
            if prediction[i] != label and truth[i] != label:
                matrix[1][1] += 1 # tn
            if prediction[i] != label and truth[i] == label:
                matrix[0][1] += 1 # fn
            if prediction[i] == label and truth[i] != label:
                matrix[1][0] += 1 # fp
            return matrix
    
    def micro_f1(matrix):
        denominator = 2 * matrix[0][0] + matrix[1][0] + matrix[0][1]
        if denominator == 0:
            return 0.0
        else:
            return (2 * matrix[0][0] / denominator)

    for c in classes:
        c_matrix = confusion_matrix(truth,prediction,c)
        sum_micro_f1 += micro_f1(c_matrix)
    return (sum_micro_f1/len(classes))


# In[22]:


def prediction_result(scores):
    """
    with an input of scores from the model, find the largest one and find its label index. 
    """
    precdction = [int(np.argmax(scores[i])) for i in range(scores.shape[0])]
    return precdction


# In[23]:


# train the model
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 20
tokens, labels = train_tokens, train_labels

batches = [(tokens[i],labels[i]) for i in range(len(tokens)) 
           if tokens[i] == '.']

for epoch in range(1, num_epochs+1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)
    running_loss = 0.0

    for t, l in batches:
#     for t, l in tqdm.tqdm(batches):
        model.zero_grad()
        # work with input tokens from train dataset
        batch_token2idx = [train_token2idx.get(ele, 0) for ele in t]
        inp = torch.tensor(batch_token2idx, dtype=torch.long)
        # to keep the number of dimensions 
        inp = torch.unsqueeze(inp, 1)
        # to filter nan 
        inp = torch.where(torch.isnan(inp), torch.full_like(inp, 0), inp)
        # work with input labels from train dataset
        batch_label2idx = [train_label2idx.get(ele, 0) for ele in l]
        inp_l = torch.tensor(batch_label2idx, dtype=torch.long)
        inp_l = torch.unsqueeze(inp_l, 1)
        inp_l = torch.where(torch.isnan(inp_l), torch.full_like(inp_l, 0), inp_l)
        
        output = model(inp_l)
        inp_l = torch.squeeze(inp_l, 1)

        loss = loss_function(output,inp_l)
        loss.backward()
# finetune: gradient clipping
#         torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
#                                            max_norm = 3, norm_type=2)
        optimizer.step()
        # calculate loss per epoch
        prediction = prediction_result(output.detach().numpy())

        running_loss += loss.item() * inp.size(0)

    epoch_loss = running_loss / len(batches)

    print(f"Loss per epoch - {epoch_loss}")
    
    print("Validating on dev test: ")
    # validate the model on dev dataset in each epoch
    with torch.no_grad():
        # dev dataset
        dev_token2idx = [train_token2idx.get(token, 0) for token in dev_tokens]
        inputs = torch.tensor(dev_token2idx, dtype=torch.long)
        dev_label2idx = [train_label2idx.get(label, 0) for label in dev_labels]

        inputs = torch.unsqueeze(inputs, 1)
        inputs = torch.where(torch.isnan(inputs), torch.full_like(inputs, 0), inputs)

        outputs = model(inputs)
        prediction = prediction_result(outputs)
        dev_macro_F1 = macro_f1(labels, dev_labels, prediction)
        print(f"Macro F1 score on dev：{dev_macro_F1}")
        print('-' * 10)
        
    
    


# In[24]:


print("Evaluating on test: ")
# evaluate the model on test set
with torch.no_grad():
    test_token2idx = [train_token2idx.get(token, 0) for token in test_tokens]
    inputs = torch.tensor(test_token2idx, dtype=torch.long)
    test_label2idx = [train_label2idx.get(label, 0) for label in test_labels]
            
    inputs = torch.unsqueeze(inputs, 1)
    inputs = torch.where(torch.isnan(inputs), torch.full_like(inputs, 0), inputs)

    outputs = model(inputs)
    test_prediction = prediction_result(outputs)
    test_macro_F1 = macro_f1(labels, test_labels, test_prediction)
    print(f"Macro F1 score on test：{test_macro_F1}")
        


# In[ ]:




