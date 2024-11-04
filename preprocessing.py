import collections

def dataset_loader(filename):
    """
    takes blank line as a seprator and gets sentences;
    takess tab("\t") as a seperatoe and get tokens and correspoding pos tags for each sentencs
    returns a list of sentences and a list of labels
    """
    sentences = []
    labels_of_sentences = []
    with open(filename, 'r', encoding = "utf-8") as f:
        tokens = []
        labels = []
        for l in f:
        # for i, l in enumerate(f): # for debugging
        #     if i < 120: # for debugging
            if l.startswith('-DOCSTART-'):
                continue
            if l != "\n":
                token, pos, chunk, ner = l.strip().split("\t")
                token = token.lower()
                tokens.append(token)
                labels.append(ner)
            else:
                if len(tokens) > 0:
                    sentences.append(tokens)
                    labels_of_sentences.append(labels)
                    tokens = []
                    labels = []
    return sentences, labels_of_sentences

def transform_token2idx(sentences):
    """
    indexes the vocabulary of a dataset
    """

    token2idx = {}
    token2idx["<UNK>"] = len(token2idx)
    token2idx["<PAD>"] = len(token2idx)
    tokens_freq = collections.Counter([t for s in sentences for t in s])
    for t,c in tokens_freq.items():
        if c >= 1:
            token2idx[t] = len(token2idx)
    return token2idx

def sentence_indexing(sentences, token2idx):
    """
    converts sentences from a list of tokens to a list of indices of tokens
    """
    indexed_sentences = []
    for s in sentences:
        indexed_tokens = []
        for token in s:
            indexed_tokens.append(token2idx.get(token, token2idx["<UNK>"]))
        indexed_sentences.append(indexed_tokens)
    return indexed_sentences
#  classes of this NER sequence tagging task
label2idx = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC':4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, 'O': 8}

def label_indexing(labels_of_sentences, label2idx):
    """
    converts labels of sentences from a list of labels to a list of indices of labels
    """
    indexed_labels_of_sent = []
    for labels in labels_of_sentences:
        indexed_labels = []
        for l in labels:
            indexed_labels.append(label2idx[l])
        indexed_labels_of_sent.append(indexed_labels)
    return indexed_labels_of_sent

