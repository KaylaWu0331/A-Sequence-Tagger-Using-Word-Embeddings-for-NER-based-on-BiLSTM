# A Sequence Tagger Using Word Embeddings for NER based on BiLSTM
## Project Description:
This project presents the implementation of a sequence tagger for Named Entity Recognition (NER) using word embeddings, leveraging a bidirectional Long Short-Term Memory (BiLSTM) network. The model architecture consists of a BiLSTM layer with a single 100-dimensional hidden unit.

## Key Features of the Model
- Utilizes cross-entropy loss as the loss function and the Adam optimizer for model training.
- Trained over 20 epochs with a batch size of 1.

## Environment Setup
The project has been developed using the following environment:
- Python 3.6
- Third-Party Library: PyTorch, Numpy, Pandas

## Data Description:
The dataset used in this project is pre-split into training, development, and test sets. The input tokens are provided in the first column, and the corresponding labels are presented in the last column.

## Word Embedding:
I adopt [Global Vectors for Word Representation(GloVe)](https://nlp.stanford.edu/projects/glove/) in word embeddings. Specifically, pre-trained word vectors derived from Wikipedia 2014 and Gigaword 5 are utilized (6B tokens, 400K vocab, uncased, 50d vectors). The embeddings can be downloaded from [this link](https://nlp.stanford.edu/data/glove.6B.zip).

## Result:
The model performance is evaluated by recording the macro-averaged F1 scores on the development set across all 20 epochs, as well as the final macro-averaged F1 score on the test set. The results are as follows:
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

