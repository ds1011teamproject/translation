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
