#!/usr/bin/env bash

# fetching the training data
cd data
wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
tar -xvzf training-parallel-europarl-v7.tgz
rm training-parallel-europarl-v7.tgz
