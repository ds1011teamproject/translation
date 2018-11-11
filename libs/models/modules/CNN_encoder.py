import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.data_loaders import IwsltLoader
from libs.models.modules import fasttext_loader
from libs.data_loaders.IwsltLoader import PAD_IDX


class Encoder(nn.Module):
    """
    MT Encoder
    Convolutional neural network (CNN)
    """

    def __init__(self, vocab_size, emb_size, hidden_size,
                 kernel_size, dropout_prob):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            padding_idx=PAD_IDX,
        )

        # # Load and copy pre-trained embedding weights
        # weights = fasttext_loader.create_weights(IwsltLoader.id2token)
        # self.embedding.weight.data.copy_(torch.from_numpy(weights))
        # self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            emb_size, hidden_size, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, enc_input, lengths):
        """
        """
        bsz, seq_len = enc_input.shape

        x = self.embedding(enc_input)
        x = x.transpose(1, 2)  # Conv1d expects (bsz x features x length)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool1d(h, seq_len).squeeze(2)  # Max pooling, drop dim=2
        h = self.dropout(h)  # Dropout regularization

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)

        return h.unsqueeze(0)
