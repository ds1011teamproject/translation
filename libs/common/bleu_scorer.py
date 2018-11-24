from sacrebleu import (
    corpus_bleu,
    DEFAULT_TOKENIZER,
)


class BLEUScorer():

    def __init__(self):
        pass

    def bleu(self, true, pred, score=True):
        """
        Calculate BLEU score using sacreblue

        @param true: True translation
        @param pred: Predicted translation
        @param score: Return score only

        cf. https://github.com/mjpost/sacreBLEU
        """
        bleu = corpus_bleu(
            true,
            pred,
            # (defaults)
            smooth="exp",
            smooth_floor=0.0,
            force=False,
            lowercase=False,
            tokenize=DEFAULT_TOKENIZER,
            use_effective_order=False,
        )
        if score:
            return bleu.score
        else:
            return bleu


if __name__ == "__main__":
    # Demo
    scorer = BLEUScorer()

    true = """Hello, how are you?
    I'm fine thank you."""
    pred = """Hello, how are you?
    I'm fine thank you."""
    print(scorer.bleu(true, pred, score=False))
    """
    BLEU(score=100.00000000000004,
    counts=[11, 10, 9, 8], totals=[11, 10, 9, 8],
    precisions=[100.0, 100.0, 100.0, 100.0],
    bp=1.0, sys_len=11, ref_len=11)
    """

    true = """Hello, how are you?
    I'm fine thank you."""
    pred = """Hi, how are you?
    I'm fine thanks."""
    print(scorer.bleu(true, pred, score=False))
    """
    BLEU(score=59.00468726392806,
    counts=[8, 6, 5, 4], totals=[11, 10, 9, 8],
    precisions=[72.72727272727273, 60.0, 55.55555555555556, 50.0],
    bp=1.0, sys_len=11, ref_len=10)
    """

    true = """Hello, how are you?
    I'm fine thank you."""
    pred = """Hi, what's up?
    Not much."""
    print(scorer.bleu(true, pred, score=False))
    """
    BLEU(score=4.9323515694897075,
    counts=[3, 0, 0, 0], totals=[11, 10, 9, 8],
    precisions=[27.272727272727273, 5.0, 2.7777777777777777, 1.5625],
    bp=1.0, sys_len=11, ref_len=8)
    """
