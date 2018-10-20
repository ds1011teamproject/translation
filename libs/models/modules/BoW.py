"""
custom nn Module class defn
"""
import torch
import torch.nn as nn


class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """

    def __init__(self, vocab_size, emb_dim):
        """
        :param vocab_size: size of the vocabulary.
        :param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, 2)

    def forward(self, data, length):
        """

        :param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        :param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0], 1).expand_as(out).float()

        # return logits
        out = self.linear(out.float())
        return out
