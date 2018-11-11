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
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_directions, dropout_prob):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
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


class AttnDecoder(nn.Module):
    """
    Machine Translation Decoder
    Gated recurrent unit (GRU)
    """
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_directions, dropout_prob, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length))
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)
        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True if num_directions == 2 else False
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, dec_in, hidden):
        # embedding
        embedded = self.embed(dec_in)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = self.out(output).squeeze(1)
        output = self.softmax(output)   # logits
        return output, hidden, attn_weights
