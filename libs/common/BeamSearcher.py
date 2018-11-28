"""
lib for performing beam search over a predefined width
"""
from libs.data_loaders.IwsltLoader import EOS_IDX
import logging
import torch
logger = logging.getLogger('__main__')


def beam_search(init_dec_in, enc_results, decoder, target_length, beam_width=3):
    """
    returns the largest probable result
    :param init_dec_in: inital decoder state
    :param enc_results: inital encoder hidden or tuple of (enc_out, hidden), depending on model
    :param decoder: decoder model reference
    :param target_length: the length of the target tensor in time steps
    :param beam_width: int width of the beam
    :return: returns the predicted results (list of words in a batch)
    """

    # running through the decoder for the first time
    dec_out = decoder(init_dec_in, enc_results)
    topv, topi = dec_out.topk(beam_width)  # keep the pruned hypothesis in topv
    hist_stack = topi.clone().unsqueeze(2)  # use the 3rd dimension to track time
    eos_stack = (topi.clone().zero_().unsqueeze(2) + 1).float()  # turns to 0 when we encounter the first EOS
    first_eos_filter = (topi != EOS_IDX).float()
    eos_stack = eos_stack * first_eos_filter.unsqueeze(2)

    # each column is a prior hypotheis, loop to create the beam_width ** 2 new hypothesis
    for i in range(target_length - 1):
        cur_hypo_stack = None
        cur_hist_stack = None
        cur_eos_stack = None
        last_step = (i == target_length - 2)

        for j in range(topv.shape[1]):  # looping through the hypothesis list
            cur_hypo_proba = topv[:, j].unsqueeze(1)
            next_dec_in = topi[:, j].unsqueeze(1)
            dec_out = decoder(next_dec_in, enc_results)
            topv_next, topi_next = dec_out.topk(beam_width)

            # computing the cumulative probabilities
            prev_eos_mask = eos_stack[:, :, -1]
            cur_eos_mask = (topi_next != EOS_IDX).float()  # creating the mask for EOS
            next_eos_mask = cur_eos_mask * prev_eos_mask
            cprob_next = cur_hypo_proba + next_eos_mask * topv_next

            # tracking eos history
            prev_eos_hist = torch.cat([eos_stack[:, j, :].unsqueeze(1)] * beam_width, 1)
            new_eos_layer = torch.cat([prev_eos_hist, next_eos_mask.unsqueeze(2)], 2)

            # tracking token index history
            prev_token_layer = torch.cat([hist_stack[:, j, :].unsqueeze(1)] * beam_width, 1)
            new_stack = torch.cat([prev_token_layer, topi_next.unsqueeze(2)], 2)

            # --- expanding to the beam_width **2 ---
            if cur_hist_stack is None:  # TOKEN INDEX
                cur_hist_stack = new_stack
            else:
                cur_hist_stack = torch.cat((cur_hist_stack, new_stack), dim=1)
            if cur_hypo_stack is None:  # TOKEN INDEX
                cur_hypo_stack = cprob_next
            else:
                cur_hypo_stack = torch.cat((cur_hypo_stack, cprob_next), dim=1)
            if cur_eos_stack is None:  # TOKEN INDEX
                cur_eos_stack = new_eos_layer
            else:
                cur_eos_stack = torch.cat((cur_eos_stack, new_eos_layer), dim=1)
            # ---------------------------------------

        # adjusting the cur_hypo_stack to account for length and eos
        hypo_lengths = cur_eos_stack.sum(dim=2)
        length_penalties = length_penalty(hypo_lengths)
        eval_matrix = cur_hypo_stack / length_penalties

        # searching for the max probabilities in the the beam_width ** 2 possibilities
        if last_step:
            _, topi_res = eval_matrix.topk(1)
        else:
            _, topi_res = eval_matrix.topk(beam_width)

        topv = torch.gather(cur_hypo_stack, 1, topi_res)

        # select the right cur_hist_stack and cur_eos_stack
        hist_stack = select_based_on_index(cur_hist_stack, topi_res)
        eos_stack = select_based_on_index(cur_eos_stack, topi_res)

    masked_hist = (hist_stack * eos_stack.long()).squeeze(1)
    predicted = []
    for k in range(masked_hist.shape[0]):
        cur_word_tsr = masked_hist[k, :]
        cur_sent = []
        for l in range(cur_word_tsr.shape[0]):
            cur_word = cur_word_tsr[l]
            if cur_word == 0:  # either pad, or masked
                break
            cur_sent.append(cur_word.item())
        cur_sent.append(EOS_IDX)
        predicted.append(cur_sent)
    return predicted


def beam_search_attn(init_dec_in, hidden, enc_out, decoder, target_length, beam_width=3):
    """
    returns the largest probable result
    :param init_dec_in: inital decoder state
    :param enc_results: inital encoder hidden or tuple of (enc_out, hidden), depending on model
    :param decoder: decoder model reference
    :param target_length: the length of the target tensor in time steps
    :param beam_width: int width of the beam
    :return: returns the predicted results (list of words in a batch)
    """

    # running through the decoder for the first time
    dec_out, hidden = decoder(init_dec_in, hidden, enc_out)
    topv, topi = dec_out.topk(beam_width)  # keep the pruned hypothesis in topv
    hist_stack = topi.clone().unsqueeze(2)  # use the 3rd dimension to track time
    eos_stack = (topi.clone().zero_().unsqueeze(2) + 1).float()  # turns to 0 when we encounter the first EOS
    first_eos_filter = (topi != EOS_IDX).float()
    eos_stack = eos_stack * first_eos_filter.unsqueeze(2)

    # each column is a prior hypotheis, loop to create the beam_width ** 2 new hypothesis
    for i in range(target_length - 1):
        cur_hypo_stack = None
        cur_hist_stack = None
        cur_eos_stack = None
        last_step = (i == target_length - 2)

        for j in range(topv.shape[1]):  # looping through the hypothesis list
            cur_hypo_proba = topv[:, j].unsqueeze(1)
            next_dec_in = topi[:, j].unsqueeze(1)
            dec_out, hidden = decoder(next_dec_in, hidden, enc_out)
            topv_next, topi_next = dec_out.topk(beam_width)

            # computing the cumulative probabilities
            prev_eos_mask = eos_stack[:, :, -1]
            cur_eos_mask = (topi_next != EOS_IDX).float()  # creating the mask for EOS
            next_eos_mask = cur_eos_mask * prev_eos_mask
            cprob_next = cur_hypo_proba + next_eos_mask * topv_next

            # tracking eos history
            prev_eos_hist = torch.cat([eos_stack[:, j, :].unsqueeze(1)] * beam_width, 1)
            new_eos_layer = torch.cat([prev_eos_hist, next_eos_mask.unsqueeze(2)], 2)

            # tracking token index history
            prev_token_layer = torch.cat([hist_stack[:, j, :].unsqueeze(1)] * beam_width, 1)
            new_stack = torch.cat([prev_token_layer, topi_next.unsqueeze(2)], 2)

            # --- expanding to the beam_width **2 ---
            if cur_hist_stack is None:  # TOKEN INDEX
                cur_hist_stack = new_stack
            else:
                cur_hist_stack = torch.cat((cur_hist_stack, new_stack), dim=1)
            if cur_hypo_stack is None:  # TOKEN INDEX
                cur_hypo_stack = cprob_next
            else:
                cur_hypo_stack = torch.cat((cur_hypo_stack, cprob_next), dim=1)
            if cur_eos_stack is None:  # TOKEN INDEX
                cur_eos_stack = new_eos_layer
            else:
                cur_eos_stack = torch.cat((cur_eos_stack, new_eos_layer), dim=1)
            # ---------------------------------------

        # adjusting the cur_hypo_stack to account for length and eos
        hypo_lengths = cur_eos_stack.sum(dim=2)
        length_penalties = length_penalty(hypo_lengths)
        eval_matrix = cur_hypo_stack / length_penalties

        # searching for the max probabilities in the the beam_width ** 2 possibilities
        if last_step:
            _, topi_res = eval_matrix.topk(1)
        else:
            _, topi_res = eval_matrix.topk(beam_width)

        topv = torch.gather(cur_hypo_stack, 1, topi_res)

        # select the right cur_hist_stack and cur_eos_stack
        hist_stack = select_based_on_index(cur_hist_stack, topi_res)
        eos_stack = select_based_on_index(cur_eos_stack, topi_res)

    masked_hist = (hist_stack * eos_stack.long()).squeeze(1)
    predicted = []
    for k in range(masked_hist.shape[0]):
        cur_word_tsr = masked_hist[k, :]
        cur_sent = []
        for l in range(cur_word_tsr.shape[0]):
            cur_word = cur_word_tsr[l]
            if cur_word == 0:  # either pad, or masked
                break
            cur_sent.append(cur_word.item())
        cur_sent.append(EOS_IDX)
        predicted.append(cur_sent)
    return predicted


def select_based_on_index(unfiltered_tsr, idx_tsr):
    """ selects the right parts of a unfiltered tensor based on the provided index, batch first """
    hist_stack_wip = []
    for i in range(idx_tsr.shape[0]):
        top_idx = idx_tsr[i, :]
        cur_res = unfiltered_tsr[i, :, :].index_select(0, top_idx)
        hist_stack_wip.append(cur_res.unsqueeze(0))
    return torch.cat(hist_stack_wip, 0)


def length_penalty(sequence_lengths, penalty_factor=0.65):
    """ from https://arxiv.org/abs/1609.08144 """
    return (5 + sequence_lengths.float() ** penalty_factor) / (5. + 1.) ** penalty_factor
