import os

import pandas as pd


class Vocabulary:
    def __init__(self, tokens):
        self.tokens = tokens
        self.id_tokens = {
            token: idx
            for idx, token in enumerate(tokens)
        }

    def __len__(self):
        return len(self.tokens)

    def itos(self, idx):
        return self.tokens[idx]

    def stoi(self, token):
        return self.id_tokens[token]


def create_vocab_flickr30k(path):
    df = pd.read_csv(os.path.join(path, "results_20130124.token"), header=None, sep='\t')

    captions = df[1].to_list()
    tokenized_sentences = [sentence.split() for sentence in captions]
    all_tokens = [token for sentence in tokenized_sentences for token in sentence]

    all_tokens = set(all_tokens)
    all_tokens = sorted(all_tokens)
    all_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + all_tokens

    return Vocabulary(all_tokens)
