import numpy as np
from preprocessing import label2idx, dataset_loader, transform_token2idx, sentence_indexing, label_indexing
def embedding_loader(filename):
    """
    loads glove.6B.50d.txt pretrained word embeddings as a indexed token dictionary, 
    including '<UNK>':0 and '<PAD>':1, and list of embedding vectors, 
    where the vector for '<UNK>' is zero vetor and for 'PAD' is the average vector. 
    """
    indexed_tokens = {}
    vecs = []
    indexed_tokens["<UNK>"] = len(indexed_tokens)
    indexed_tokens["<PAD>"] = len(indexed_tokens)
    with open(filename, encoding='utf-8') as f:
        for l in f:
            values = l.split()
            token = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            indexed_tokens[token] = len(indexed_tokens)
            vecs.append(embedding)
    embeddings_list = np.array(vecs)
    unk_embedding = embeddings_list.mean(axis=0)
    embeddings_list = np.vstack([unk_embedding, embeddings_list])
    pad_embedding = np.zeros(shape=embeddings_list.shape[-1])
    embeddings_list = np.vstack([pad_embedding, embeddings_list])

    return indexed_tokens, embeddings_list

glove_file = './glove.6B.50d/glove.6B.50d.txt'
indexed_tokens, embeddings_list = embedding_loader(glove_file)

def embeddings_matrix_generator(vocab, indexed_tokens, embeddings_list):
    """
    takes pretrained word embedding list, indexed token dictionary,and the vocabulary of training dataset, 
    returns an embedding matix, which gives tokens in the vocabulary the pretrained embedding vetors.
    """
    embeddings_matrix = np.zeros((len(vocab), 50))
    for token, idx in vocab.items():
        if token in indexed_tokens:
            embedding_vector = embeddings_list[indexed_tokens[token]]
            embeddings_matrix[idx] = embedding_vector
        else:
            embedding_vector = embeddings_list[indexed_tokens['<UNK>']]
            embeddings_matrix[idx] = embedding_vector
    return embeddings_matrix
# load training dataset to get its vocabulary
sentences_train, labels_of_sentences_train = dataset_loader("./data/train.conll")
vocab = transform_token2idx(sentences_train)
# To avoid redundancy, I get training data and lebels for training here.
indexed_sentences_train = sentence_indexing(sentences_train, vocab)
indexed_labels_of_sent_train = label_indexing(labels_of_sentences_train, label2idx)

embeddings_matrix = embeddings_matrix_generator(vocab, indexed_tokens, embeddings_list) 

