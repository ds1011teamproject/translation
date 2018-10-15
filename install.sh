#!/usr/bin/env bash

# making the directory tree
mkdir data
mkdir data/training/
mkdir models

python setup.py install clean

# fetching the training data
cd data/training/
wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
tar -xvzf training-parallel-europarl-v7.tgz

