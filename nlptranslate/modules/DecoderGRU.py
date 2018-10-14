import torch.nn as nn

from nlptranslate.config import basic_settings

config = basic_settings.default


class DecoderGRU(nn.Module):
    """
    Decoder
    Gated recurrent unit (GRU)
    """
    def __init__(self, input_size,
                 embedding_size=config["embedding_size"],
                 hidden_size=config["hidden_size"]):
        super(DecoderGRU, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden_state = self.gru(embedded, hidden)
        output = output.view(1, output.size(2))
        linear = self.linear(output)
        softmax = self.softmax(linear)
        return output, softmax, hidden_state
