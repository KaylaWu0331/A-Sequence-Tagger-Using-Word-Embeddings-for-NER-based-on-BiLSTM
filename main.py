import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np

import multiprocessing

from embeddings import indexed_sentences_train, indexed_labels_of_sent_train, embeddings_matrix, vocab
from preprocessing import label2idx, dataset_loader, sentence_indexing, label_indexing
from models import ConllDataset, collate_fn, BiLSTM
from params import lr,epochs, clip_grad_norm
# load dev dataset as indexed_labels_of_sent_dev and indexed_sentences_dev for ConllDataset
sentences_dev, labels_of_sentences_dev = dataset_loader("./data/dev.conll")
indexed_sentences_dev = sentence_indexing(sentences_dev, vocab)
indexed_labels_of_sent_dev = label_indexing(labels_of_sentences_dev, label2idx)
# load test dataset as indexed_labels_of_sent_test and indexed_sentences_test for ConllDataset
sentences_test, labels_of_sentences_test = dataset_loader("./data/test.conll")
indexed_sentences_test = sentence_indexing(sentences_test, vocab)
indexed_labels_of_sent_test = label_indexing(labels_of_sentences_test, label2idx)
# instantiate ConllDataset and get 3 ConllDataset instances for 3 datasets
train_dataset = ConllDataset(indexed_sentences_train, indexed_labels_of_sent_train)
dev_dataset = ConllDataset(indexed_sentences_dev, indexed_labels_of_sent_dev)
test_dataset = ConllDataset(indexed_sentences_test, indexed_labels_of_sent_test)

torch.manual_seed(0)
# instatiate Dataloader with ConllDataset and collate_fn
# set train_dataloader in parallel mode to improve the efficiency of training
train_dataloader = DataLoader(dataset= train_dataset, batch_size=1, shuffle = True, collate_fn = collate_fn, num_workers=2, pin_memory= True)
dev_dataloader = DataLoader(dataset= dev_dataset, batch_size=1, shuffle = True, collate_fn = collate_fn)
test_dataloader = DataLoader(dataset= test_dataset, batch_size=1, shuffle = True,collate_fn = collate_fn)

# instatiate BiLSTM model with embeddings_matrix and indexed labels(classes) 
model = BiLSTM(vocab, torch.tensor(embeddings_matrix), label2idx)
#  use CrossEntropy-loss as criterion to calculate loss
criterion = nn.CrossEntropyLoss(ignore_index= -1, reduction="mean")
# use Adam to update weights and minimize loss
optimizer = torch.optim.Adam(params = model.parameters(), lr = lr, weight_decay = 0, amsgrad = False)
epochs = epochs

def get_predition_result(dataloader):
    """
    when evaluating the model, takes development or test dataset, returns list of predictions and total loss,  
    also convert dataloader back to list of labels in order to calculate F1 scores and determine early stopping
    """
    pred_list = []
    true_list = []
    total_loss = 0.0   
    for tokens, labels in dataloader:
        ops = model(tokens)
        loss = criterion(ops.float(), labels.view(-1).long())
        total_loss += loss.item()
        mask = labels != -1
        o_labels = labels[mask].tolist()
        true_list.append(o_labels)
        pred = []
        for i in range(len(o_labels)):
            _, max_index = torch.max(ops[i], dim=0)
            pred.append(max_index.item())

        pred_list.append(pred)

    return pred_list, true_list, total_loss

def f1_score(predicted_list, true_list, classes):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for c in classes:
        tp = predicted_list.count(c) and true_list.count(c)
        fp = predicted_list.count(c) and not true_list.count(c)
        fn = true_list.count(c) and not predicted_list.count(c)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return f1_score

def macro_f1_score(predicted_lists, true_lists, classes):
    f1_scores = []
    for i in range(len(predicted_lists)):
        f1_scores.append(f1_score(predicted_lists[i], true_lists[i], classes))
    macro_f1_score = sum(f1_scores) / len(f1_scores)
    return macro_f1_score

if __name__ == '__main__':
    multiprocessing.freeze_support() # to support parallel mode in train_dataloader
    for e in range(epochs):
        print(f"Epoch {e+1} starts...")
        print("Training the model with training dataset ...")
        model.train()
        for i, (tokens,labels) in enumerate(train_dataloader):
            ops = model(tokens)
            loss = criterion(ops.float(), labels.view(-1).long())
            loss.backward()
            clip_grad_norm_( # gradient clipping to prevent exploding gradients
                model.parameters(),
                max_norm=clip_grad_norm,
                norm_type=2,
            )
            if i % 2 == 0: # indicates the parallel mode in train_dataloader, 2 is the number of threads
                optimizer.step()
                optimizer.zero_grad()
            if i % 100 == 0: # report loss for every 100 batches
                print(f'Epoch: {e+1}, Batch: {i}, Loss: {loss.item():.4f}')

        
        print("Evaluating the model with development detaset...") 
        model.eval()
        best_loss = np.Inf # initialize variables for early stopping
        patience = 3
        counter = 0
        with torch.no_grad():
            dev_pred, dev_labels, dev_total_loss = get_predition_result(dev_dataloader)
            # report macro f1 score on dev dataset per epoch
            print(f"The {e+1}th epoch: The marco F1 score on dev dataset: ",macro_f1_score(dev_pred, dev_labels, label2idx.values()))
            # early stopping: if there is no better average of losses appears for 3 epochs, then stop training the model and save the current params
            val_loss = dev_total_loss/len(dev_labels)
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'model_params.pt')
        else: 
            counter += 1
        if counter >= patience:
            break

    print("Evaluating the model with test detaset...")  
    model.load_state_dict(torch.load('model_params.pt'))
    model.eval()
    with torch.no_grad():
        test_pred, test_labels, _ = get_predition_result(test_dataloader)
        # report macro f1 score on test dataset
        print("After 20 epochs: The marco F1 score on test dataset: ",macro_f1_score(test_pred, test_labels, label2idx.values()))

# outputs:
# The 1th epoch: The marco F1 score on dev dataset:  0.875115188277595
# The 2th epoch: The marco F1 score on dev dataset:  0.8751151882775963
# The 3th epoch: The marco F1 score on dev dataset:  0.8751151882775963
# The 4th epoch: The marco F1 score on dev dataset:  0.9455040082007662
# The 5th epoch: The marco F1 score on dev dataset:  0.9659872071652431
# The 6th epoch: The marco F1 score on dev dataset:  0.971780397991584
# The 7th epoch: The marco F1 score on dev dataset:  0.9709544486084003
# The 8th epoch: The marco F1 score on dev dataset:  0.9798581224122912
# The 9th epoch: The marco F1 score on dev dataset:  0.9823135754744201
# The 10th epoch: The marco F1 score on dev dataset:  0.981496515103009
# The 11th epoch: The marco F1 score on dev dataset:  0.982756415301488
# The 12th epoch: The marco F1 score on dev dataset:  0.9829753374146527
# The 13th epoch: The marco F1 score on dev dataset:  0.9839912207603118
# The 14th epoch: The marco F1 score on dev dataset:  0.9838887313863081
# The 15th epoch: The marco F1 score on dev dataset:  0.9837004460519823
# The 16th epoch: The marco F1 score on dev dataset:  0.9839470012633746
# The 17th epoch: The marco F1 score on dev dataset:  0.9836441221747071
# The 18th epoch: The marco F1 score on dev dataset:  0.9845460759268119
# The 19th epoch: The marco F1 score on dev dataset:  0.9844836976947042
# The 20th epoch: The marco F1 score on dev dataset:  0.9841089627577841

# After 20 epochs: The marco F1 score on test dataset:  0.9678993265667207


