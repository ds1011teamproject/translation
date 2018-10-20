class PathKey:
    TEST_PATH = 'test_path'
    TRAIN_PATH = 'train_path'
    MODEL_SAVES = 'model_saves'


class HyperParamKey:
    TRAIN_PLUS_VAL_SIZE = 'train_plus_val_size'
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


class LoaderParamKey:
    ACT_VOCAB_SIZE = 'act_vocab_size'


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

