
## Trial 1

I ran the basic training model for 1 iteration and wanted to demonstrate the save functionality

**hope you all find this useful!**


Hyperparameters used:
num_epochs - 1
lr - 0.01
train_plus_val_size - 25000
test_size - 25000
val_size - 5000
voc_size - 100000
train_loop_check_freq - 100
embedding_dim - 50
batch_size - 32
ngram_size - 2
remove_punc - True
check_early_stop - True
es_look_back - 5
es_req_prog - 0.01
optim_enc - <class 'torch.optim.adam.Adam'>
optim_dec - <class 'torch.optim.adam.Adam'>
scheduler - <class 'torch.optim.lr_scheduler.ExponentialLR'>
scheduler_gamma - 0.95
criterion - <class 'torch.nn.modules.loss.CrossEntropyLoss'>

Loader parameters used:
act_vocab_size - 100002

Control parameters used:
save_best_model - True
save_each_epoch - True
test_path - data/aclImdb/test/
train_path - data/aclImdb/train/
model_saves - model_saves/
model_path - model_saves/scratch/