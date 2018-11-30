import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.data_loaders import IwsltLoader
from libs.data_loaders.IwsltLoader import PAD_IDX
from config.basic_conf import DEVICE


class Encoder(nn.Module):
    """
    MT Encoder
    Convolutional neural network (CNN)
    """

    def __init__(self, vocab_size, emb_size, hidden_size, kernel_size, dropout_prob,
                 trained_emb=None, freeze_emb=False, use_attn=False):
        super(Encoder, self).__init__()

        # self.vocab_size = vocab_size
        # self.emb_size = emb_size
        self.hidden_size = hidden_size
        # self.kernel_size = kernel_size
        # self.dropout_prob = dropout_prob
        self.use_attn = use_attn

        pad_size = kernel_size // 2
        # embedding layer
        if trained_emb is None:
            self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        else:
            trained_emb = torch.from_numpy(trained_emb).float()
            self.embed = nn.Embedding.from_pretrained(trained_emb, freeze=freeze_emb)

        # cnn layers
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size=kernel_size, padding=pad_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=pad_size)

        # non-linear layers to generate context vector
        self.gen_ctx = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, enc_input, lengths):
        # enc_input: batch * max_len
        batch_size, seq_len = enc_input.size()
        # embed
        enc_input = self.embed(enc_input)

        # ====== old ======
        # # 1st layer
        # enc_input = self.conv1(enc_input.transpose(1, 2)).transpose(1, 2)
        # enc_input = F.relu(enc_input.contiguous().view(-1, enc_input.size(-1))).view(batch_size, seq_len, enc_input.size(-1))
        # # 2nd layer
        # enc_input = self.conv2(enc_input.transpose(1, 2)).transpose(1, 2)
        # enc_input = F.relu(enc_input.contiguous().view(-1, enc_input.size(-1))).view(batch_size, seq_len, enc_input.size(-1))
        # ===== refactor =====
        enc_input = F.relu(self.conv1(enc_input.transpose(1, 2)))
        enc_input = F.relu(self.conv2(enc_input))

        # max-pooling
        # ======= old =========
        # mp = nn.MaxPool2d((seq_len, 1))
        # context = mp(enc_input).reshape(batch_size, self.hidden_size)  # hidden = torch.squeeze(mp(hidden), dim=1)
        # ===== borrowed from CNN_encoder.py =====
        context = F.max_pool1d(enc_input, seq_len).squeeze(2)

        # generate context vector
        context = self.gen_ctx(context).unsqueeze(0)

        # return
        if self.use_attn:
            print('enc_hid_vectors:', enc_input.size(), 'context_vec:', context.size())
            return enc_input, context
        else:
            return context


class EncoderTry(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, kernel_size, seq_len, dropout_prob,
                 trained_emb=None, freeze_emb=False, use_attn=False):
        super(EncoderTry, self).__init__()

        # self.vocab_size = vocab_size
        # self.emb_size = emb_size
        self.hidden_size = hidden_size
        # self.kernel_size = kernel_size
        # self.dropout_prob = dropout_prob
        self.use_attn = use_attn

        pad_size = kernel_size // 2
        # embedding layer
        if trained_emb is None:
            self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        else:
            trained_emb = torch.from_numpy(trained_emb).float()
            self.embed = nn.Embedding.from_pretrained(trained_emb, freeze=freeze_emb)

        # cnn layers
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size=kernel_size, padding=pad_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=pad_size)

        # compress hidden vecs
        self.cpr_linear = nn.Linear(seq_len, 1)

        # non-linear layers to generate context vector
        self.gen_ctx = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, enc_input, lengths):
        # enc_input: batch * max_len
        batch_size, seq_len = enc_input.size()
        # embed
        enc_input = self.embed(enc_input)

        # cnn layers
        # # 1st layer
        # enc_input = self.conv1(enc_input.transpose(1, 2)).transpose(1, 2)
        # enc_input = F.relu(enc_input.contiguous().view(-1, enc_input.size(-1))).view(batch_size, seq_len, enc_input.size(-1))
        # # 2nd layer
        # enc_input = self.conv2(enc_input.transpose(1, 2)).transpose(1, 2)
        # enc_input = F.relu(enc_input.contiguous().view(-1, enc_input.size(-1))).view(batch_size, seq_len, enc_input.size(-1))

        enc_input = F.relu(self.conv1(enc_input.transpose(1, 2)))
        enc_input = F.relu(self.conv2(enc_input))
        # size (batch * hidden * seq_len)

        # Replace max-pooling layer
        context = torch.tanh(self.cpr_linear(enc_input)).squeeze(2)  # todo:  check F.relu
        # size (batch * hidden) after .squeeze(2)

        # generate context vector
        context = self.gen_ctx(context).unsqueeze(0)
        print('=== context vec ===', context.size())

        # return
        if self.use_attn:
            return enc_input.transpose(1, 2), context
        else:
            return context
