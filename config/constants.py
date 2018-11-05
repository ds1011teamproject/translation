class PathKey:
    DATA_PATH_VI = 'data_path_vi'
    DATA_PATH_ZH = 'data_path_zh'
    INPUT_LANG = 'input_lang'
    OUTPUT_LANG = 'output_lang'
    MODEL_SAVES = 'model_saves'
    MODEL_PATH = 'model_path'


class HyperParamKey:
    TRAIN_PLUS_VAL_SIZE = 'train_plus_val_size'
    LR = 'lr'
    TEST_SIZE = 'test_size'
    VAL_SIZE = 'val_size'
    NUM_EPOCH = 'num_epochs'
    EMBEDDING_DIM = 'embedding_dim'
    NGRAM_SIZE = 'ngram_size'
    REMOVE_PUNC = 'remove_punc'
    BATCH_SIZE = 'batch_size'
    VOC_SIZE = 'voc_size'
    TRAIN_LOOP_EVAL_FREQ = 'train_loop_check_freq'
    CHECK_EARLY_STOP = 'check_early_stop'
    EARLY_STOP_LOOK_BACK = 'es_look_back'
    EARLY_STOP_REQ_PROG = 'es_req_prog'
    OPTIMIZER_ENCODER = 'optim_enc'
    OPTIMIZER_DECODER = 'optim_dec'
    SCHEDULER = 'scheduler'
    SCHEDULER_GAMMA = 'scheduler_gamma'
    CRITERION = 'criterion'


class ControlKey:
    SAVE_BEST_MODEL = 'save_best_model'
    SAVE_EACH_EPOCH = 'save_each_epoch'


class LoaderParamKey:
    ACT_VOCAB_SIZE = 'act_vocab_size'


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

