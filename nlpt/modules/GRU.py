import numpy as np
import torch
from torch import nn
import torch.optim as optim
from nlpt.modules.EncoderGRU import EncoderGRU
from nlpt.modules.DecoderGRU import DecoderGRU
from nlpt.config import basic_conf as conf


class GRU(object):
    def __init__(self, input_size, output_size):
        super(GRU, self).__init__()

        self.encoder = EncoderGRU(input_size).to(conf.DEVICE)
        self.decoder = DecoderGRU(output_size).to(conf.DEVICE)

        self.loss = nn.CrossEntropyLoss()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())

        sos, eos = torch.LongTensor(1, 1).to(conf.DEVICE).zero_(), torch.LongTensor(1, 1).to(conf.DEVICE).zero_()
        sos[0, 0], eos[0, 0] = 0, 1

        self.sos, self.eos = sos, eos

    def train(self, _input, target):
        target.insert(0, self.sos)
        target.append(self.eos)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encoder
        hidden_state = self.encoder.first_hidden()
        for ivec in _input:
            _, hidden_state = self.encoder.forward(ivec, hidden_state)

        # Decoder
        total_loss, outputs = 0, []
        for i in range(len(target) - 1):
            _, softmax, hidden_state = self.decoder.forward(target[i], hidden_state)

            outputs.append(np.argmax(softmax.data.cpu().numpy(), 1)[:, np.newaxis])
            total_loss += self.loss(softmax, target[i][0])

        total_loss /= len(outputs)
        total_loss.backward()

        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        # use total_loss.data[0] for version 0.3.0_4 and below,
        # .item() for 0.4.0
        return total_loss.item(), outputs

    def eval(self, _input):
        hidden_state = self.encoder.first_hidden()

        # Encoder
        for ivec in _input:
            _, hidden_state = self.encoder.forward(ivec, hidden_state)

        sentence = []
        _input = self.sos

        # Decoder
        while _input.data[0, 0] != 1:
            output, _, hidden_state = self.decoder.forward(_input, hidden_state)
            word = np.argmax(output.data.numpy()).reshape((1, 1))
            _input = torch.LongTensor(word).to(conf.DEVICE)
            sentence.append(word)

        return sentence

    def save(self):
        torch.save(self.encoder.state_dict(), conf.gru_encoder_model)
        torch.save(self.decoder.state_dict(), conf.gru_decoder_model)

    def cuda(self):
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        return self
