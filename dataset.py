import errno
import os
import requests
import torch

DATA_PATH = 'data'


class SifDataset:
    def __init__(self, path=DATA_PATH):
        self.path = path
        self.train = self.load('train')
        self.dev = self.load('dev')


    def download(self, split, full_path):
        try:
            os.makedirs(self.path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        url = "https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-" + split
        print('Downloading split {} from {}'.format(split, url))
        r = requests.get(url, stream=True)
        with open(full_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)

    def load(self, split):
        full_path = os.path.join(self.path, 'sentiment-' + split)
        if not os.path.isfile(full_path):
            self.download(split, full_path)

        print('Loading split {} from {}'.format(split, full_path))
        result = []
        with open(full_path) as f:
            for line in f:
                sentence, label = line.split('\t')
                label = int(label)
                result.append((sentence, label))
        return result


def preprocess_dataset(dataset, dictionary):
    exs = []
    labels = []
    freq = {}
    total = 0
    for sentence, label in dataset:
        tokenized = preprocess_sentence(sentence, dictionary)
        for token in tokenized:
            if token in freq.keys():
                freq[token] += 1
            else:
                freq[token] = 1
            total += 1
        if tokenized.numel() > 0:
            exs.append(tokenized)
            labels.append(torch.LongTensor([label]))
    for token, n in freq.items():
        freq[token] = float(n) / float(total)
    all_tokens = torch.LongTensor([int(token) for token in freq.keys()])
    freq_tensor = torch.FloatTensor(all_tokens.max() + 1)
    for token in all_tokens:
        freq_tensor[token] = freq[token]
    return exs, labels, freq_tensor

def preprocess_sentence(sentence, dictionary):
    return torch.LongTensor([dictionary[w] for w in sentence.split() if w in dictionary])
