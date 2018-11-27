class PathKey:
    DATA_PATH = 'data_path'
    INPUT_LANG = 'input_lang'
    OUTPUT_LANG = 'output_lang'
    MODEL_SAVES = 'model_saves'
    MODEL_PATH = 'model_path'


class HyperParamKey:
    # vocabulary parameters
    USE_FT_EMB = 'use_ft_embeddings'
    FREEZE_EMB = 'freeze_emb_vectors'
    VOC_SIZE = 'voc_size'
    REMOVE_PUNC = 'remove_punc'
    # model parameters
    EMBEDDING_DIM = 'embedding_dim'
    HIDDEN_SIZE = 'hidden_size'
    ENC_NUM_LAYERS = 'enc_layers'
    ENC_NUM_DIRECTIONS = 'enc_directions'
    DEC_NUM_LAYERS = 'dec_layers'
    DEC_NUM_DIRECTIONS = 'dec_directions'
    ENC_DROPOUT = 'enc_dropout'
    DEC_DROPOUT = 'dec_dropout'
    KERNEL_SIZE = 'kernel_size'
    MAX_LENGTH = 'max_length'
    # train related
    BATCH_SIZE = 'batch_size'
    NUM_EPOCH = 'num_epochs'
    TRAIN_LOOP_EVAL_FREQ = 'train_loop_check_freq'
    TEACHER_FORCING_RATIO = 'teacher_forcing_ratio'
    BEAM_SEARCH_WIDTH = 'beam_search_width'
    CHECK_EARLY_STOP = 'check_early_stop'
    EARLY_STOP_LOOK_BACK = 'es_look_back'
    EARLY_STOP_REQ_PROG = 'es_req_prog'
    NO_IMPROV_LOOK_BACK = 'ni_look_back'
    NO_IMPROV_LR_DECAY = 'ni_lr_decay'
    OPTIMIZER = 'optim_method'
    ENC_WEIGHT_DECAY = 'enc_weight_decay'
    DEC_WEIGHT_DECAY = 'dec_weight_decay'
    ENC_LR = 'enc_lr'
    DEC_LR = 'dec_lr'
    SCHEDULER = 'scheduler'
    SCHEDULER_GAMMA = 'scheduler_gamma'
    CRITERION = 'criterion'
    NUM_TRAIN_SENT_TO_LOAD = 'num_train_sent_to_load'


class ControlKey:
    SAVE_BEST_MODEL = 'save_best_model'
    SAVE_EACH_EPOCH = 'save_each_epoch'


class LoaderParamKey:
    ACT_VOCAB_SIZE = 'act_vocab_size'
    TRAINED_EMB = 'trained_embeddings'


class StateKey:
    MODEL_STATE = 'model_state'
    OPTIM_STATE = 'optim_state'
    SCHED_STATE = 'sched_state'
    ITER_CURVES = 'iter_curves'
    EPOCH_CURVES = 'epoch_curves'
    HPARAMS = 'hparams'
    LPARAMS = 'lparams'
    CPARAMS = 'cparams'
    CUR_EPOCH = 'cur_epoch'
    LABEL = 'label'
    META = 'meta'


class LoadingKey:
    LOAD_CHECKPOINT = 'checkpoint'
    LOAD_BEST = 'best'


class OutputKey:
    BEST_VAL_ACC = 'best_val_acc'
    BEST_VAL_LOSS = 'best_val_loss'
    FINAL_VAL_ACC = 'final_val_acc'
    FINAL_VAL_LOSS = 'final_val_loss'
    FINAL_TRAIN_ACC = 'final_train_acc'
    FINAL_TRAIN_LOSS = 'final_train_loss'


# Reference: nltk.corpus.stopwords.fileids()
class Language:
    ARA = 'arabic'
    DAN = 'danish'
    DUT = 'dutch'
    ENG = 'english'
    FIN = 'finnish'
    FRE = 'french'
    GER = 'german'
    GRE = 'greek'
    HUN = 'hungarian'
    IND = 'indonesian'
    ITA = 'italian'
    KAZ = 'kazakh'
    NEP = 'nepali'
    NOR = 'norwegian'
    POR = 'portuguese'
    ROM = 'romanian'
    RUS = 'russian'
    SPA = 'spanish'
    SWE = 'swedish'
    TUR = 'turkish'


LogConfig = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(levelname)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'filters': {},
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': None,  # to be override
            'formatter': 'standard',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'default'],
            'level': 'INFO',
            'propagate': True
        }
    }
}


class FastText:
    DATA_PATH = 'data/wiki-news-300d-1M.vec'
