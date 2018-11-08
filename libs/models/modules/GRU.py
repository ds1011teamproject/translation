import torch
import torch.nn as nn

from libs.data_loaders.IwsltLoader import PAD_IDX


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

    def forward(self, enc_input, lengths):
        # init hidden
        hidden = enc_input.new_zeros(self.num_layers * self.num_directions,
                                     enc_input.size()[0],  # batch_size
                                     self.hidden_size)
        # embedding
        embedded = self.embed(enc_input)
        # rnn
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.numpy(), batch_first=True)
        output, hidden = self.gru(embedded, hidden)
        # we don't need the output vectors thus no 'pad_packed_sequence'
        return hidden


class Decoder(nn.Module):
    """
    Machine Translation Decoder
    Gated recurrent unit (GRU)
    """
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_directions, dropout_prob,):
        super(Decoder, self).__init__()
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
            bidirectional=True if num_directions == 2 else False
        )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, dec_in, enc_out, dec_in_lens):
        # init hidden
        hidden = dec_in.new_zeros(self.num_layers * self.num_directions,
                                  dec_in.size()[0],
                                  self.hidden_size)
        # embedding
        embedded = self.embed(dec_in)
        # rnn
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, dec_in_lens.numpy(), batch_first=True)
        output, hidden = self.gru(embedded, hidden)
        # print("after GRU:", output.size())
        output = self.out(output[0])
        # print("after linear:", output.size())
        output = self.softmax(output)   # logits
        return output, hidden
