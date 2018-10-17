import torch
import torch.nn as nn

from config import basic_hparams, basic_conf as conf

config = basic_hparams.DEFAULT_HPARAMS


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

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, 1, -1)
        output, hidden_state = self.gru(embedded, hidden)
        return output, hidden_state

    def first_hidden(self):
        return torch.FloatTensor(1, 1, self.hidden_size).to(conf.DEVICE).zero_()
