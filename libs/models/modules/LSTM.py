import torch
import torch.nn as nn

from config import basic_hparams, basic_conf as conf
from config.constants import HyperParamKey

config = basic_hparams.DEFAULT_HPARAMS


class Encoder(nn.Module):
    """
    Encoder
    Long Short Term Memory (LSTM)
    """
    def __init__(self, input_size,
                 embedding_size=config[HyperParamKey.EMB_SIZE],
                 hidden_size=config[HyperParamKey.HIDDEN_SIZE]):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        # todo: change to nn.LSTM
        self.gru = nn.GRU(embedding_size, hidden_size, 1)

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, 1, -1)
        output, hidden_state = self.gru(embedded, hidden)
        return output, hidden_state

    # todo: adapt first_hidden for LSTM
    def first_hidden(self):
        return torch.FloatTensor(1, 1, self.hidden_size).to(conf.DEVICE).zero_()


class Decoder(nn.Module):
    """
    Decoder
    Long Short Term Memory (LSTM)
    """
    def __init__(self, input_size,
                 embedding_size=config[HyperParamKey.EMB_SIZE],
                 hidden_size=config[HyperParamKey.HIDDEN_SIZE]):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        # todo: change to nn.LSTM
        self.gru = nn.GRU(embedding_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, _input, hidden):
        embedded = self.embedding(_input)
        output, hidden_state = self.gru(embedded, hidden)
        output = output.view(1, output.size(2))
        linear = self.linear(output)
        softmax = self.softmax(linear)
        return output, softmax, hidden_state
