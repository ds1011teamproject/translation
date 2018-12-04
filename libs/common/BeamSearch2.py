"""
Xialiang's Beam Search

"""
from libs.data_loaders.IwsltLoader import EOS_IDX, SOS_IDX
import logging
import torch
from config.basic_conf import DEVICE
logger = logging.getLogger('__main__')

"""
Length penalty methods:
  - SIMPLE: divide log probability by number of tokens in translated sentence
            treat short/long translations equally.
  - FACTOR: divide log probability by result of function len_penalty (Rong)
            give preference to shorter sentences. 
"""
SIMPLE = 'simple'
FACTOR = 'factor'


def len_penalty(lens, fac=0.65):
    return (5 + lens ** fac) / (5. + 1.) ** fac


def beam_search(init_dec_in, hidden, enc_out, decoder, max_len, beam_width=3,
                len_penalty_type=FACTOR):
    """
    I use the same signature as Rong's beam-search method, painless if we switch later
    :param init_dec_in:
    :param hidden: last hidden state of encoder
    :param enc_out: context vectors from encoder
    :param decoder: model decoder
    :param max_len: max length of sentences
    :param beam_width: search width
    :param len_penalty_type: method of length penalty, 'simple' or 'factor'
    :return: list of indices with highest probability
    """

    finished = []
    beam = [[] for _ in range(max_len)]
    beam[0] = [(0, hidden, [SOS_IDX])]

    for t in range(1, max_len):  # seq_len axis:
        for log_prob, h_state, pred_temp in beam[t - 1]:
            curr_tok_idx = pred_temp[-1]
            dec_in = torch.LongTensor([curr_tok_idx]).unsqueeze(1).to(DEVICE)
            dec_out, h_state = decoder(dec_in, h_state, enc_out)
            topv, topi = dec_out.topk(beam_width)
            prob = topv[0].cpu().numpy()
            idxs = topi[0].cpu().numpy()

            for i in range(beam_width):
                p = prob[i]
                idx = idxs[i]
                if idx == EOS_IDX:
                    finished.append((log_prob + p, pred_temp + [idx]))
                else:
                    beam[t].append((log_prob + p, h_state, pred_temp + [idx]))
        # prune beam[t]
        beam[t].sort(key=lambda tup: tup[0])
        beam[t] = beam[t][-beam_width:]
    # sort finished
    if len_penalty_type == SIMPLE:
        finished.sort(key=lambda tup: tup[0] / len(tup[1]))
    else:
        finished.sort(key=lambda tup: tup[0] / len_penalty(len(tup[1])))
    return finished[-1][1] if finished != [] else []






