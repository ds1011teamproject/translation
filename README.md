# Neural machine translation

Team project for [Natural Language Processing with Representation Learning
(DS-GA 1011)](https://docs.google.com/document/d/1o0TTWocbkqPa9qsTCXnEFXf3NZzwZLLLSw7SSZmNla8/edit#heading=h.ga92jtl8vlih)

## Data

[[website] ACL 2014 NINTH WORKSHOP ON STATISTICAL MACHINE TRANSLATION](http://www.statmt.org/wmt14/index.html)

[[website] Shared Task: Machine Translation](http://www.statmt.org/wmt14/translation-task.html)

[[download] Europarl v7](http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz)

[[download] Common Crawl corpus](http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz)

### Download on HPC

```
$ cd /scratch/<netID>/
$ wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
$ tar -xvzf training-parallel-europarl-v7.tgz
training/europarl-v7.cs-en.cs
training/europarl-v7.cs-en.en
training/europarl-v7.de-en.de
training/europarl-v7.de-en.en
training/europarl-v7.es-en.en
training/europarl-v7.es-en.es
training/europarl-v7.fr-en.en
training/europarl-v7.fr-en.fr
```

**!** Don't forget to update the data file paths in `settings.py`.

## Requirements

```
$(HPC) module load anaconda3/5.3.0
$ conda create -n mt python=3.6
$ conda activate mt
(mt)$ conda install pytorch torchvision -c pytorch
(mt)$ python main.py
```

See [this guide for detailed instructions on how to run on HPC](https://github.com/mvishwali28/quantifier-rnn-learning).

## RNN encoder-decoder

PyTorch implementation of recurrent neural network (RNN) encoder-decoder architecture model for statistical machine translation, cf. ["Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"](https://arxiv.org/pdf/1406.1078.pdf) (Cho et al., 2014)

### Further references

[pytorch/fairseq/models/LSTM](https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py)

**NB:** Forward of the encoder produces (batch size x sequence length x hidden state), which is the input to the decoder that also takes in the previous output tokens.

