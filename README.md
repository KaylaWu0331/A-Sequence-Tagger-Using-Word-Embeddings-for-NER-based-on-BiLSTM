# A Sequence Tagger Using Word Embeddings for NER based on BiLSTM
## Project Description:
In this project, I first implement a simple sequence tagger using word embeddings for named entity recognition, where the architecture is a bidirectional LSTM with a single 100-dimensional hidden layer.
Furthermore, I use the following parameters in the training phase:
- Crossentropy-loss and the Adam optimizer
- Train for 20 Epochs
- Set the batch-size to 1

## Environment:
- Python 3.6
- Third-Party Library: PyTorch, Numpy, Pandas

## Data Description:
The data is already split into a train, dev, and test set. The input tokens are specified in the first column and the labels are in last column.

## Word Embedding:
I adopt [Global Vectors for Word Representation(GloVe)](https://nlp.stanford.edu/projects/glove/) in word embeddings. Specifically, the pre-trained word vectors are selected from Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d vectors). Here is [the download URL](https://nlp.stanford.edu/data/glove.6B.zip).

## Result:
I record the macro-averaged F1 scores on the dev data (for all 20 epochs) and the macro-averaged F1 scores of the final model on the test data.
```Python
outputs:
The 1th epoch: The marco F1 score on dev dataset:  0.875115188277595
The 2th epoch: The marco F1 score on dev dataset:  0.8751151882775963
The 3th epoch: The marco F1 score on dev dataset:  0.8751151882775963
The 4th epoch: The marco F1 score on dev dataset:  0.9455040082007662
The 5th epoch: The marco F1 score on dev dataset:  0.9659872071652431
The 6th epoch: The marco F1 score on dev dataset:  0.971780397991584
The 7th epoch: The marco F1 score on dev dataset:  0.9709544486084003
The 8th epoch: The marco F1 score on dev dataset:  0.9798581224122912
The 9th epoch: The marco F1 score on dev dataset:  0.9823135754744201
The 10th epoch: The marco F1 score on dev dataset:  0.981496515103009
The 11th epoch: The marco F1 score on dev dataset:  0.982756415301488
The 12th epoch: The marco F1 score on dev dataset:  0.9829753374146527
The 13th epoch: The marco F1 score on dev dataset:  0.9839912207603118
The 14th epoch: The marco F1 score on dev dataset:  0.9838887313863081
The 15th epoch: The marco F1 score on dev dataset:  0.9837004460519823
The 16th epoch: The marco F1 score on dev dataset:  0.9839470012633746
The 17th epoch: The marco F1 score on dev dataset:  0.9836441221747071
The 18th epoch: The marco F1 score on dev dataset:  0.9845460759268119
The 19th epoch: The marco F1 score on dev dataset:  0.9844836976947042
The 20th epoch: The marco F1 score on dev dataset:  0.9841089627577841

After 20 epochs: The marco F1 score on test dataset:  0.9678993265667207
```

