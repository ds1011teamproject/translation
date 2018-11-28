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
        self.hidden_size = hidden_size // num_directions
        if trained_emb is None:
            self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        else:
            trained_emb = torch.from_numpy(trained_emb).float()
            self.embed = nn.Embedding.from_pretrained(trained_emb, freeze=freeze_emb)
        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=(True if num_directions == 2 else False)
        )

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers * self.num_directions, batch_size,
                            self.hidden_size), device=DEVICE)

    def forward(self, enc_input, lengths):
        # init hidden
        batch_size = enc_input.size()[0]
        hidden = self.init_hidden(batch_size)
        # embedding
        enc_input = self.embed(enc_input)
        # rnn
        enc_input = nn.utils.rnn.pack_padded_sequence(enc_input, lengths, batch_first=True)
        _, hidden = self.gru(enc_input, hidden)
        # we don't need the output vectors thus no 'pad_packed_sequence'
        if self.num_directions == 2:
            hidden = hidden[-2:].transpose(0, 1).contiguous().view(1, batch_size, -1)
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
        # for init_hidden
        self.init_linear = nn.Linear(hidden_size, hidden_size)
        # for compare y_hat(t-1) and context vector
        self.compare = nn.Linear(emb_size + hidden_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True if num_directions == 2 else False
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # (Cho 2014) Learning Phrase Rep using RNN encoder-decoder for SMT
    def init_hidden(self, enc_out):
        hidden = self.init_linear(enc_out)
        return torch.tanh(hidden)

    def forward(self, dec_in, hidden, enc_out):
        dec_in = self.embed(dec_in)

        # compare y_hat(t-1) and context vector
        dec_in = torch.cat((dec_in, enc_out.transpose(0, 1)), dim=2)
        # dec_in: batch * 1 * (emb + hidden_size)
        dec_in = self.compare(dec_in)
        # batch_size * 1 * hidden_size
        dec_in = F.relu(dec_in)
        dec_in, hidden = self.gru(dec_in, hidden)
        dec_in = self.out(dec_in).squeeze(1)
        dec_in = self.softmax(dec_in)  # logits
        return dec_in, hidden
