### Version 0.3.0 - 10/17/2018
Changes:
- further restructure of training models and modules by algorithm
- add default logging setup and write to file

Comments:
- discuss log file save
- my next to-do is to implement RNN with LSTM and improve the abstraction of RNN 
- A simple note: *please concatenate the sentences into one log message, rather than multiple lines, when they are highly related*

### Version 0.2.1 - 10/16/2018
Changes:
- add argparse usage in main enter point (for data files and configuration files)
- reformat config files (basic_conf/basic_hparams) with pre-defined keys

Comments:
- discuss user input config file format with team, require some design of hyperparmeter tuning process.

### Version 0.2.0 - 10/15/2018
Changes:
- more reorganization, got rid of the installer, it felt more clunky than useful
- main was gutted and mostly moved to ModelManager.py
- Added a basic registry for models so that we can just call ModelManager.set_model('GRU') for initializing the model
    - created a super class for models called TranslationModel that enforces the interface between GRU and ModelManager
- wrote down many todos
- enabled notebook support, see main_nb.ipynb, initialize ModelManager with extra parameter tqdm_mode='notebook'
- added some examples of logging usage in main and in ModelManager.py
    - init function is in the config.basic_conf.py, should only call this 1x
- touched up all of the readme

Comments:
- we're now at the point that we can start working independantly and contributing code in parallel, but I hope we can over-communicate nonetheless!

### Version 0.1.1 - 10/15/2018
Changes:
- Bug fix in GRU.py to the training loop

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
