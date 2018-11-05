# Neural machine translation

Team project for [Natural Language Processing with Representation Learning
(DS-GA 1011)](https://docs.google.com/document/d/1o0TTWocbkqPa9qsTCXnEFXf3NZzwZLLLSw7SSZmNla8/edit#heading=h.ga92jtl8vlih)

## Data


### * Project Data
Vietnum-English and Chinese-English parallel corpus provided by the instructors. 

## Installation
Do this installation if you are going to experiment with the code
```
$ git clone https://github.com/ds1011teamproject/translation.git
$ mkdir data
$ mkdir model_saves
```

**!** If you are using different folders for data and models, update the data file paths in `nlpt.config.basic_conf.py`.


## Releasing Updates:
Please do the following items when pushing a change out
- increment version on setup.py
- add change notes to changelogs/README.md


## Running on HPC

```
$ module load anaconda3/5.3.0  # HPC only
$ module load cuda/9.0.176 cudnn/9.0v7.0.5  # HPC only
$ conda create -n mt python=3.6
$ conda activate mt
$ conda install torch pandas numpy tqdm
```

See [this guide for detailed instructions on how to run on HPC](https://github.com/mvishwali28/quantifier-rnn-learning).

On HPC, you might need to add the following line to your `~/.bashrc`:

```
. /share/apps/anaconda3/5.3.0/etc/profile.d/conda.sh
```

## Running Locally
This will execute the version that is installed in site-packages
```
$ python -m main
```

## Running in a notebook
See main_nb.ipynb

## RNN encoder-decoder

PyTorch implementation of recurrent neural network (RNN) encoder-decoder architecture model for statistical machine translation, cf. ["Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"](https://arxiv.org/pdf/1406.1078.pdf) (Cho et al., 2014)

### Further references

[pytorch/fairseq/models/LSTM](https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py)

**NB:** Forward of the encoder produces (batch size x sequence length x hidden state), which is the input to the decoder that also takes in the previous output tokens.
