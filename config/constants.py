class PathKey:
    INPUT_LANG = 'input_lang'
    OUTPUT_LANG = 'output_lang'
    ENC_SAVE = 'encoder_out'
    DEC_SAVE = 'decoder_out'
    RESULT_SAVE = 'results_df'


class HyperParamKey:
    VOC_SIZE = "vocab_size"
    MAX_LEN = "max_length"
    EMB_SIZE = "embedding_size"
    HIDDEN_SIZE = "hidden_size"
    NUM_BATCH = "num_batches"
    NUM_EPOCH = "num_epochs"


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
            'format': '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'  # ,
            # 'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'filters': {},
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
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
            'handlers': ['default', 'console'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

