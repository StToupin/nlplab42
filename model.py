import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter



class BowModel(nn.Module):
    def __init__(self, emb_tensor, freq):
        super(BowModel, self).__init__()
        n_embedding, dim = emb_tensor.size()
        self.embedding = nn.Embedding(n_embedding, dim, padding_idx=0)
        self.embedding.weight = Parameter(emb_tensor, requires_grad=False)
        self.out = nn.Linear(dim, 2)
        self.freq = Parameter(freq, requires_grad=False)

    def forward(self, input):
        '''
        input is a [batch_size, sentence_length] tensor with a list of token IDs
        '''
        embedded = self.embedding(input)
        freq = self.freq
        weights = 1. / freq[input[0,:].data]
        # bow = embedded.mean(dim = 1)
        # print(bow)
        bow = torch.mm(weights.unsqueeze(0), embedded[0,:,:]).squeeze(1) / input.size(1)
        bow /= 1.e5
        # print(bow)
        return F.log_softmax(self.out(bow))


# input [1, 7]
# weights [7]
# embedded [1, 7, 300]
# bow [1, 300]