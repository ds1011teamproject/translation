import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.data_loaders.IwsltLoader import PAD_IDX
from config.basic_conf import DEVICE


class Encoder(nn.Module):
    """
    Machine Translation Encoder
    Gated recurrent unit (GRU)
    """
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_directions,
                 dropout_prob, trained_emb=None, freeze_emb=False):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        if trained_emb is None:
            self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        else:
            trained_emb = torch.from_numpy(trained_emb).float()
            self.embed = nn.Embedding.from_pretrained(trained_emb, freeze=freeze_emb)
        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=(True if num_directions == 2 else False)
        )

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers * self.num_directions, batch_size,
                            self.hidden_size // self.num_directions), device=DEVICE)

    def forward(self, enc_input, lengths):
        # init hidden
        hidden = self.init_hidden(enc_input.size()[0])
        # embedding
        embedded = self.embed(enc_input)
        # rnn
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        output, hidden = self.gru(embedded, hidden)
        # we don't need the output vectors thus no 'pad_packed_sequence'
        return hidden


class Decoder(nn.Module):
    """
    Machine Translation Decoder
    Gated recurrent unit (GRU)
    """
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_directions,
                 dropout_prob, trained_emb=None, freeze_emb=False):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        if trained_emb is None:
            self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        else:
            trained_emb = torch.from_numpy(trained_emb).float()
            self.embed = nn.Embedding.from_pretrained(trained_emb, freeze=freeze_emb)
        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True if num_directions == 2 else False
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, dec_in, hidden):
        # embedding
        embedded = self.embed(dec_in)
        # todo: do we need a ReLU like lab8?
        # embedded = F.relu(embedded)
        # rnn
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output).squeeze(1)
        output = self.softmax(output)   # logits
        return output
