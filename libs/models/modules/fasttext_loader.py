import io
import numpy as np

from config.constants import FastText, HyperParamKey


def load_vectors(fp, id2tok):
    """
    Load fastText embeddings for vocabulary words
    """
    toks = set(id2tok)
    fin = io.open(fp, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        if line.split()[0] in toks:
            tokens = line.rstrip().split(" ")
            data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def create_weights(id2tok):
    """
    Create weights matrix from pre-trained embeddings
    """
    # Load vectors
    fasttext = load_vectors(FastText.DATA_PATH, id2tok)

    # Initialize empty weights matrix
    weights = np.zeros((HyperParamKey.VOC_SIZE, HyperParamKey.EMBEDDING_DIM))
    found = 0  # Found tokens [15365]

    for i, tok in enumerate(id2tok):
        try:
            weights[i] = fasttext[tok]
            found += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.6, size=(
                HyperParamKey.EMBEDDING_DIM, ))

    return weights
