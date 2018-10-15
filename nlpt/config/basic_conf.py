"""
non hyperparameter settings
"""

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

en_path = u"data/training/europarl-v7.fr-en.en"
fr_path = u"data/training/europarl-v7.fr-en.fr"

gru_encoder_model = u"models/gru_encoder.pkl"
gru_decoder_model = u"models/gru_decoder.pkl"
