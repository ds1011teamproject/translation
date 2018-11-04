"""
child class implementation that loads the imdb data

I hacked this up as an example of an implementation of a data loader handler, it is pretty messy ...

The idea is for new projects to implement their own loaders
"""
from libs.data_loaders.BaseLoader import BaseLoader
import numpy as np
import torch
from torch.utils.data import DataLoader
from config.constants import HyperParamKey, PathKey, LoaderParamKey
from config.basic_conf import DEVICE
from torch.utils.data import Dataset
from collections import Counter
import os
import logging
import string

logger = logging.getLogger('__main__')
punctuations = string.punctuation
PAD_TOKEN, PAD_IDX = '<pad>', 0
UNK_TOKEN, UNK_IDX = '<unk>', 1


class ImdbLoader(BaseLoader):
    def __init__(self, io_paths, hparams, tqdm):
        super().__init__(io_paths, hparams, tqdm)
        pass

    def load(self):
        self._load_raw_data()
        self._data_to_pipe()
        return {LoaderParamKey.ACT_VOCAB_SIZE: len(self.data['vocab'])}

    def _load_raw_data(self):
        # just gets the data, doesn't implement pickling
        train_and_val_set = construct_dataset(self.io_paths[PathKey.DATA_PATH] + 'train/'
                                              , self.hparams[HyperParamKey.TRAIN_PLUS_VAL_SIZE])

        test_set = construct_dataset(self.io_paths[PathKey.DATA_PATH] + 'test/'
                                     , self.hparams[HyperParamKey.TEST_SIZE])

        logger.info("extracting ngrams from train/val set...")
        train_and_val_data = extract_ngrams(train_and_val_set,
                                            self.hparams[HyperParamKey.NGRAM_SIZE],
                                            self.tqdm,
                                            remove_punc=self.hparams[HyperParamKey.REMOVE_PUNC])

        logger.info("extracting ngrams from test set...")
        test_data = extract_ngrams(test_set,
                                   self.hparams[HyperParamKey.NGRAM_SIZE],
                                   self.tqdm,
                                   remove_punc=self.hparams[HyperParamKey.REMOVE_PUNC])

        train_ngram_indexer, _ = create_ngram_indexer(train_and_val_data,
                                                      self.tqdm,
                                                      topk=self.hparams[HyperParamKey.VOC_SIZE],
                                                      val_size=self.hparams[HyperParamKey.VAL_SIZE])

        logger.info("setting each dataset's token indexes")
        train_and_val_data = process_dataset_ngrams(train_and_val_data, train_ngram_indexer)
        test_data = process_dataset_ngrams(test_data, train_ngram_indexer)

        self.data['train'] = train_and_val_data[:len(train_and_val_data) - self.hparams[HyperParamKey.VAL_SIZE]]
        self.data['val'] = train_and_val_data[-self.hparams[HyperParamKey.VAL_SIZE]:]
        self.data['test'] = test_data
        self.data['vocab'] = train_ngram_indexer

    def _data_to_pipe(self):
        """
        coverts the data objects to the torch.*.DataLoader pipes
        """
        imdb_train = IMDBDataset(self.data['train'])
        imdb_validation = IMDBDataset(self.data['val'])
        imdb_test = IMDBDataset(self.data['test'])

        self.loaders['train'] = DataLoader(dataset=imdb_train,
                                           batch_size=self.hparams[HyperParamKey.BATCH_SIZE],
                                           collate_fn=imdb_collate_func,
                                           shuffle=True)

        self.loaders['val'] = DataLoader(dataset=imdb_validation,
                                         batch_size=self.hparams[HyperParamKey.BATCH_SIZE],
                                         collate_fn=imdb_collate_func,
                                         shuffle=False)

        self.loaders['test'] = DataLoader(dataset=imdb_test,
                                          batch_size=self.hparams[HyperParamKey.BATCH_SIZE],
                                          collate_fn=imdb_collate_func,
                                          shuffle=False)

        logger.info("All DataLoader pipe ready: train, val, test in ModelManager.loader.loaders (dict)")


class IMDBDatum:
    """
    Class that represents a train/validation/test datum
    - self.raw_text
    - self.label: 0 neg, 1 pos
    - self.file_name: dir for this datum
    - self.tokens: list of tokens
    - self.token_idx: index of each token in the text
    """

    def __init__(self, raw_text, label, file_name):
        self.raw_text = raw_text
        self.label = label
        self.file_name = file_name
        self.ngram = None
        self.token_idx = None
        self.tokens = None

    def set_ngram(self, ngram_ctr):
        self.ngram = ngram_ctr

    def set_token_idx(self, token_idx):
        self.token_idx = token_idx

    def set_tokens(self, tokens):
        self.tokens = tokens


class IMDBDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of IMDBDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        token_idx, label = self.data_list[key].token_idx, self.data_list[key].label
        return [token_idx, len(token_idx), label]


def construct_dataset(dataset_dir, dataset_size, offset=0):
    """
    Function that loads a dataset
    :param offset: skip first offset items in this dir
    :param dataset_dir:
    :param dataset_size:
    :return: a list of IMDBDatum Objects
    """
    pos_dir = os.path.join(dataset_dir, "pos")
    neg_dir = os.path.join(dataset_dir, "neg")
    single_label_size = int(dataset_size / 2)
    output = []
    all_pos = os.listdir(pos_dir)
    all_neg = os.listdir(neg_dir)
    for i in range(offset, offset + single_label_size):
        output.append(read_file_as_datum(os.path.join(pos_dir, all_pos[i]), 1))
        output.append(read_file_as_datum(os.path.join(neg_dir, all_neg[i]), 0))
    return output


def read_file_as_datum(file_name, label):
    """
    Function that reads a file
    """
    with open(file_name, "r") as f:
        content = f.read()
        content = preprocess_text(content)
    return IMDBDatum(raw_text=content, label=label, file_name=file_name)


def preprocess_text(text):
    """
    Function that cleans the string
    """
    text = text.lower().replace("<br />", "")
    return text


def extract_ngrams(dataset, n, tqdm, remove_punc=True):
    """
    extracts the ngrams for the dataset
    :param dataset: list of IMDBDatum
    :param n: n in "n-gram"
    :param tqdm: the tqdm lib for either console or notebook
    :param remove_punc: remove all punctuations
    :return: dataset with ngrams extracted
    """
    for i in tqdm(range(len(dataset))):
        text_datum = dataset[i].raw_text
        cur_ngrams, tokens = extract_ngram_from_text(text_datum, n, remove_punc)
        dataset[i].set_ngram(cur_ngrams)
        dataset[i].set_tokens(tokens)
    return dataset


def extract_ngram_from_text(text, n, remove_punc=True):
    """
    Function that retrieves all n-grams from the input string
    :param text: raw string
    :param n: integer that tells the model to retrieve all k-gram where k<=n
    :param remove_punc: whether or not to remove punctuation from lib
    :return ngram_counter: a counter that maps n-gram to its frequency
    :return tokens: a list of parsed ngrams
    """
    tokens = text.split(" ")
    if remove_punc:
        tokens = [token.lower() for token in tokens if (token not in punctuations)]
    else:
        tokens = [token.lower() for token in tokens]

    all_ngrams = extract_all_ngrams(tokens, n)
    ngram_counter = Counter(all_ngrams)
    return ngram_counter, all_ngrams


def extract_all_ngrams(tokens, n):
    rt_list = []
    for i in range(1, n + 1):
        rt_list += ngrams(tokens, i)
    return rt_list


def ngrams(tokens, n=2):
    return zip(*[tokens[i:] for i in range(n)])


def create_ngram_indexer(dataset,
                         tqdm,
                         topk=None,
                         val_size=0):
    """
    from the dataset that has ngrams extracted, create the vocab indexer
    :param dataset: ngrams already extracted
    :param tqdm: tqdm handler depending on console or notebook
    :param topk: vocab size
    :param val_size: val_set size (to not use in the indexer)
    :return:
    """
    logger.info("constructing ngram_indexer ...")
    logger.info("indexer length %s" % len([datum.ngram for datum in dataset][:-val_size]))
    return construct_ngram_indexer([datum.ngram for datum in dataset][:-val_size], topk, tqdm)


def construct_ngram_indexer(ngram_counter_list, topk, tqdm):
    """
    Function that selects the most common topk ngrams
    index 0 reserved for <pad>
    index 1 reserved for <unk>
    :param ngram_counter_list: list of counters
    :param topk: int, # of words to keep in the vocabulary - not counting pad/unk
    :param tqdm: tqdm module initialized in ModelManager
    :return ngram2idx: a dictionary that maps ngram to an unique index
    """
    rt_dict = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    i = 2  # the index to start the rest of the tokens
    final_count = Counter()

    for elem in tqdm(ngram_counter_list):
        for key, value in elem.items():
            final_count[key] += value

    for key in dict(final_count.most_common(topk)):
        rt_dict[key] = i
        i += 1

    logger.info("final vocal size: %s" % len(rt_dict))
    return rt_dict, final_count  # length topk + 2


def process_dataset_ngrams(dataset, ngram_indexer):
    """
    processes the dataset that has ngrams already extracted
    :param dataset: list of IMDBDatum, ngrams already extracted
    :param ngram_indexer: a dictionary that maps ngram to an unique index
    :return:
    """
    for i in range(len(dataset)):
        dataset[i].set_token_idx(token_to_index(dataset[i].tokens, ngram_indexer))
    return dataset


def token_to_index(tokens, ngram_indexer):
    """
    Function that transform a list of tokens to a list of token index.
    index 0 reserved for <pad>
    index 1 reserved for <unk>
    :param tokens: list of ngram
    :param ngram_indexer: a dictionary that maps ngram to an unique index
    """
    return [ngram_indexer[token] if token in ngram_indexer else UNK_IDX for token in tokens]


def imdb_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    max_length = np.max(length_list)
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]),
                            pad_width=(0, max_length - datum[1]),
                            mode="constant", constant_values=0)
        data_list.append(padded_vec)

    """ check types on the np array """
    nparr = np.array(data_list)
    rt_tensor = torch.from_numpy(nparr).to(DEVICE)

    return [rt_tensor,
            torch.LongTensor(length_list).to(DEVICE),
            torch.LongTensor(label_list).to(DEVICE)]
