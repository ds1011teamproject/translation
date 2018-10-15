import torch
import torch.nn as nn
from torch.autograd import Variable

from nlpt.config import basic_settings

config = basic_settings.default


class EncoderGRU(nn.Module):
    """
    Encoder
    Gated recurrent unit (GRU)
    """
    def __init__(self, input_size,
                 embedding_size=config["embedding_size"],
                 hidden_size=config["hidden_size"]):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, 1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden_state = self.gru(embedded, hidden)
        return output, hidden_state

    def first_hidden(self):
        return Variable(torch.FloatTensor(1, 1, self.hidden_size).zero_())
