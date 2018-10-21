### Version 0.4.0 - 10/21/2018
Changes:
- branched the code to 'homework' with the aim of catering to using the ModelManager for the homework
- added \_\_version\_\_ to ModelManager to track current version
- replaced data with data from homework 1, the data should go into data/aclImdb/
- added BaseLoader Class as the Loader interface
- implemented ImdbLoader as a example of how the loader should be implemented
- implemented a loader registry in data_loaders
- renamed the model interface to BaseModel
- added a BagOfWords Model that implements the BaseModel
- removed the RNN model for this branch
- .gitignore now doesn't ignore the entire model_saves folder, rather only the .tar files
- New ModelManager functionality:
    - rolled up ModelManager.io_paths into ModelManager.cparams to denote "Control Parameters"
    - model saves at each epoch and new best
    - can control whether these happen using ModelManager.cparams
    - can save final models after training with a meta_string that is saved to a markdown file
    - changed the set_model method to new_model, can now take a label parameter that determines the subfolder under model_saves that the model saves files (defaults to /scratch)
    - added the training curves on the BaseModel, comes in 2 flavors:
        - 1 curve that is snapped at each model check iteration. A check iteration is each time we check for early stop, which can happen multiple times in each epoch. There is a hyperparameter that controls for this behaviour
        - 1 curve that is snapped at each model epoch
    - added a basic curve grapher: ModelManager.graph_training_curves(), see example output in model_saves/scratch/
    - implemented saving and restarting training, see use_guide.ipynb for details
    - added the ability to generate a README.MD in the model_save/<model_label>/ folder to automatically document what we were doing with each model
    - implemented a results collection routine that helps collects results from mulitple model trainings into the ModelManager.results list (which can be turned into a pd.DataFrame by ModelManager.get_results() ) 
    

Comments:
- the changes were pretty extensive, refer to use_guide.ipynb to see a demo of all the new functionality


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
