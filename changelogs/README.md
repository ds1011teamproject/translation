### Version 0.1.0 - 10/15/2018
Changes:
- reorganized package tree structure
- repackaged code into python package: nlpt
- scripted installer install.sh
- rewrote installation documentation
- rewrote execution documentation
- added dynamic GPU support for single GPU, checks if GPU available, falls back on CPU
- on the GPU each 100 steps takes roughly 10s
- refactored code to pytorch 0.4, removed torch.autograd.Variable instances
 
Comments: 
 - Their data loading and pickling routine is not great, it pickles the tensors instead of some non torch type. As a result, the pickles are locked into either cpu or cuda. If you pickle them with cpu tensors, your code will break if you try running them subsequently on cuda.
 - My next todo is to refactor the training loops into a different module so that our main.py and dev_main.py are really clean, should only contain argparse

### Version 0.0.1 - 10/14/2018
- created baseline code based on the repo: https://github.com/ShubhangDesai/rnn-encoder-decoder
- added documentaion for install and execution on HPC
