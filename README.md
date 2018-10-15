# Neural machine translation

Team project for [Natural Language Processing with Representation Learning
(DS-GA 1011)](https://docs.google.com/document/d/1o0TTWocbkqPa9qsTCXnEFXf3NZzwZLLLSw7SSZmNla8/edit#heading=h.ga92jtl8vlih)

## Data

[[website] ACL 2014 NINTH WORKSHOP ON STATISTICAL MACHINE TRANSLATION](http://www.statmt.org/wmt14/index.html)

[[website] Shared Task: Machine Translation](http://www.statmt.org/wmt14/translation-task.html)

[[download] Europarl v7](http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz)

[[download] Common Crawl corpus](http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz)

## Installation - Prod
Do this installation if you are running in prod and will not change code

Remember to uninstall if you are going back into dev mode
```
$ git clone https://github.com/ds1011teamproject/translation.git
$ bash install.sh

# if you want to download the corpus then also:
$ bash download_data.sh
```

#### Uninstall - prod
```
pip uninstall nlpt
```

## Installation - Dev
Do this installation if you are going to experiment with the code
```
$ git clone https://github.com/ds1011teamproject/translation.git
$ mkdir data
$ mkdir models
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
$ python setup.py install clean
```

See [this guide for detailed instructions on how to run on HPC](https://github.com/mvishwali28/quantifier-rnn-learning).

On HPC, you might need to add the following line to your `~/.bashrc`:

```
. /share/apps/anaconda3/5.3.0/etc/profile.d/conda.sh
```

## Running in installed Prod mode
This will execute the version that is installed in site-packages
```
$ python -m nlpt.main
#TODO: add argparse argument handling
```

## Running in uninstalled Dev mode
This will execute the version that is installed in local directory:
```
# be in the root director of the project (where main.py is located)
$ python -m dev_main
#TODO: add argparse argument handling
```

## RNN encoder-decoder

PyTorch implementation of recurrent neural network (RNN) encoder-decoder architecture model for statistical machine translation, cf. ["Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"](https://arxiv.org/pdf/1406.1078.pdf) (Cho et al., 2014)

### Further references

[pytorch/fairseq/models/LSTM](https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py)

**NB:** Forward of the encoder produces (batch size x sequence length x hidden state), which is the input to the decoder that also takes in the previous output tokens.

