### Version 0.9.0 - 11/30/2018
Changes:
- Added CNN models with/without attention:
    - to generate the context vector, before passing the hidden vectors through a 2-layer 
    connected network, I tried using max-pooling, tanh(Vh) and Relu(Vh)
- minor fix for RNN_Attention decoding method

### Version 0.8.1 - 11/28/2018
Changes:
- Added grid search demo script 
- change eval_randomly() logging level to debug 
    - report translation for randomly selected sentence during training a single model
    - depress printing during grid search

### Version 0.8.0 - 11/28/2018
Changes:
- major fix for rnn-encoder-decoder without
    - fix GRU decoder structure, Ref: Cho 2014 arXiv:1406.1078v3 
    - add init_hidden, using encoder context vector
    - refactor RNN_GRU decoding method
- simplified beam-search method, now compatible with RNN with/without attention
- misc:
    - get rid of tqdm (got error when early-stop triggered)

### Version 0.7.6 - 11/27/2018
Changes:
- switch to use BLEU for early stop
- delete outdated BLEU demo

### Version 0.7.5 - 11/27/2018
Changes:
- add beam search that works with attention

### Version 0.7.4 - 11/27/2018
Changes:
- updated model load method (see details from v0.7.3 To-do)
- updated demo files with argparse
    - train a single model: demo_SingleModel.py
    - evaluate BLEU on trained model (load and eval): demo_TrainedModelBLEU.py
    - *instruction in doc string 
- upgraded eval_model() method 
    - compatible with both training and post-train evaluation
- minor fix for beam_search

### Version 0.7.3 - 11/26/2018
Changes:
- Added BLEU score in TranslatorModel, eval_model method
    - new keys in constant OutputKey
    - new lists in iter_curves and epoch_curves
    - *later to be used for early-stop (early-stop update required)

To-do:
- update model load method, init model with read-in hparam from checkpoint
- update demo files with argparse
- add grid-search demo 

### Version 0.7.2 - 11/26/2018
Changes:
- Added beam search
    - (more details...)
- no_grad update

### Version 0.7.1 - 11/25/2018
Changes:
- Added early stop based on validation loss to training curve
- Added LR halving to training curve
- LR halving look back set to 10 iterations
- Full early stop look back set to 30 iterations
- Changed some default basic hparams

### Version 0.7.0 - 11/24/2018
Changes:
- refactor train loop and model structure
    - all translation models share the same train loop, in TranslatorModel.py
    - for each model, implement the decoding method:
        - TRAIN mode: return batch_loss as a tensor, used in train()
        - EVAL mode: return batch_loss as a float, used in compute_loss
        - TRANSLATE mode: return predicted indices, an Int list
    - *tested on NYU HPC
- update demo script

### Version 0.6.6 - 11/18/2018
Changes:
- Minor improvements:
    - Added .item() to the compute_loss fns for the RNN_GRU and RNN_GRUattn models to make memory consumption less intensive

### Version 0.6.5 - 11/17/2018
Changes:
- Added MAX_LENGTH support in IWSLT data loader
    - required for attention implementation
- Added NUM_TRAIN_SEND_TO_LOAD in IWSLT data loader, for testing training on a subset of data

### Version 0.6.4 - 11/16/2018
Changes:
- Added some small functionality to ModelManager
    - new graphing only loss functions no accuracy since irrelevant
    - added new attribute in ModelManager called self.device that shows which GPU it is on
    - training loop now pickles the self.results list to the model_saves/ at the end of each training loop (in cases where we are grid searching over many training loops)

### Version 0.6.3 - 11/16/2018
Changes:
- refactor for memory concern:
    - rename some variables in encoder/decoder and train-loop
    - ModelManager and child models don'NO_IMPROV_LOOK_BACKt keep trained-embeddings in memory, only ACT_VOC_SIZE
- updated default hyperparameters: use large hidden dimension 

### Version 0.6.2 - 11/15/2018
Changes:
- add demo script for RNN_GRU: demo_RNN_GRU.py
- support using pre-trained word vectors:
    - FastText word vectors for vi/zh/en
    - text data required in directory: DATA_PATH/word/vectors/cc.<lang>.300.vec
To-do:
- [URGENT] refactor ModelManager and codebase structure, reduce memory redundancy
- upgrade train loop for TranslatorModel

### Version 0.6.1 - 11/09/2018
Changes:
- update RNN_GRU train loop
    - add eval_randomly: translate a randomly selected sentence from given data loader
    - add compute loss: compute NLLloss on the entire data set (train/val) per report interval
       - time-consuming that slows down training, either use large report interval, or do this computation per epoch
    - update corresponding output dict etc.
    
### Version 0.6.0 - 11/08/2018
Changes:
- update constants and keys
    - seperate some hyperparameters for encoder and decoder
    - config requires specific num_layers and num_directions
    - support L2 regularization for optimizers
- refactor the base model hierarchy:
    - BaseModel: the very foundation for all deep models, provides interface/handlers to ModelManager
    - ClassifierModel: base model for classification task, inherits BaseModel, provides specific train/save/load methods
    - TranslatorModel: base model for machine translation task, inherits BaseModel, similar functionality as ClassifierModel
    - *Please implement your model for translation task inheriting TranslationModel.
- implement GRU model (vanilla version)
    - encoder/decoder structure
    - train loop, support mini-batch
    - check_early_stop on train loss history
    - add demo notebook for this model: `demo_RNN_GRU.ipynb`
    
### Version 0.5.1 - 11/04/2018
Changes:
- support minibatch training for translation task
- optimize data loading and pre-processing  

### Version 0.5.0 - 11/04/2018
Changes:
- update constants (data_path)
- update basic_config (input/output_lang for translation task)
- minor bug fix in logger initialization
- update main_nb.ipynb for project data loader demo
- add IWSLT data loader file [current]: 
    - main: load raw text to Datum list for train/val/test set
    - save (token2id, id2token, vocab_words) per language + vocab_size
    - add <EOS> at end of sentence
    - simple split and remove "\n", lower first letter in the sentence
    

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
