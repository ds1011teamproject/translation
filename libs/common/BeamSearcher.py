"""
lib for performing beam search over a predefined width
"""
from libs.data_loaders.IwsltLoader import EOS_IDX
import logging
import torch
import pandas as pd
from config.basic_conf import DEVICE
logger = logging.getLogger('__main__')


def tensor_to_df(topv, topi, prev_hidden):
    """ converts topk tensors to pairs of ([token], logprob) """
    assert topv.shape == topi.shape
    num_hypos = topv.shape[1]

    rt_dict = {}
    for i in range(num_hypos):
        rt_dict[i] = {
            'prob': topv[0, i].item(),
            'sent': [topi[0, i].item()],
            'hidden': prev_hidden.clone()
        }
    return pd.DataFrame(rt_dict).T


def combine_dfs(base_df, new_df):
    """ dfs are prob, sent dataframes """
    res_dict = {}
    i = 0
    for row in base_df.iterrows():
        base_prob = row[1]['prob']
        base_sent = row[1]['sent']

        for new_row in new_df.iterrows():
            new_prob = base_prob + new_row[1]['prob']
            new_sent = base_sent + new_row[1]['sent']
            new_hidden = new_row[1]['hidden']

            res_dict[i] = {
                'prob': new_prob,
                'sent': new_sent,
                'hidden': new_hidden
            }
            i += 1
    return pd.DataFrame(res_dict).T


def check_finished(df):
    for sent in df['sent']:
        if sent[-1] != EOS_IDX:
            return False
    return True


def beam_search(init_dec_in, hidden, enc_out, decoder, target_length, beam_width=3):
    # running through the decoder for the first time
    dec_out, prev_hidden = decoder(init_dec_in, hidden, enc_out)
    topv, topi = dec_out.topk(beam_width)  # keep the pruned hypothesis in topv

    # for storing the sentence probability pairs
    sent_probs = tensor_to_df(topv, topi, prev_hidden)

    for i in range(target_length - 1):
        # this loop is for running through the full length of the sentence
        cur_hypo_df_list = []

        for j in range(sent_probs.shape[0]):
            # this loop is to evaluate all of the possibilities
            cur_row = sent_probs[j:j + 1]
            last_val = cur_row['sent'].values[0][-1]
            if last_val != EOS_IDX:
                cur_prev_hidden = cur_row['hidden'].values[0]
                next_dec_in = torch.LongTensor([last_val]).unsqueeze(1).to(DEVICE)
                dec_out, next_hidden = decoder(next_dec_in, cur_prev_hidden, enc_out)
                topv_next, topi_next = dec_out.topk(beam_width)
                next_sp = tensor_to_df(topv_next, topi_next, next_hidden)

                cur_new_probs = combine_dfs(cur_row, next_sp)
                cur_hypo_df_list.append(cur_new_probs)
            else:
                cur_hypo_df_list.append(cur_row)

        sent_probs = pd.concat(cur_hypo_df_list, axis=0)
        # display(sent_probs)

        sent_probs = sent_probs.sort_values('prob', ascending=False)[0:beam_width]
        # display(sent_probs)

        if check_finished(sent_probs):
            break

    return sent_probs.sort_values('prob', ascending=False)['sent'].values[0]


def length_penalty(sequence_lengths, penalty_factor=0.65):
    """ from https://arxiv.org/abs/1609.08144 """
    return (5 + sequence_lengths.float() ** penalty_factor) / (5. + 1.) ** penalty_factor
