"""
DEMO: use beam search to evaluate BLEU on validation set

python demo_beam2.py \
        -p model_saves/rnn/     # path to model checkpoint files
        -m RNN_GRU              # model type
        -s best                 # 'best' or 'checkpoint'
        -w 3                    # search width
        --SIMPLE <or> --FACTOR  # length penalty method, not required, default=FACTOR

"""

import torch
import argparse

import libs.ModelManager as mm
import libs.data_loaders.IwsltLoader as iwslt
from config.constants import HyperParamKey
from config.basic_conf import DEVICE
from libs.common.BleuScorer import BLEUScorer
from libs.common.BeamSearch2 import beam_search, SIMPLE, FACTOR

parser = argparse.ArgumentParser(description="beam test")
parser.add_argument('-p', '--CKP_PATH', dest='ckp_path')
parser.add_argument('-m', '--MODEL_TYPE', dest='model_type')
parser.add_argument('-s', '--SAVE_METHOD', dest='save_method')
parser.add_argument('-w', '--WIDTH', dest='beam_width', required=True)
parser.add_argument('--SIMPLE', action='store_true')
parser.add_argument('--FACTOR', action='store_true')

args = parser.parse_args()
beam_w = int(args.beam_width)
model_type = args.model_type if args.model_type else 'RNN_Attention'
ckp_path = args.ckp_path if args.ckp_path else '/scratch/xl2053/nlp/translation/model_saves/attnESbleuSave/'
save_method = args.save_method if args.save_method else 'best'
len_penalty = SIMPLE if args.SIMPLE else FACTOR


mgr = mm.ModelManager()
bleuScorer = BLEUScorer()


# load model
mgr.load_model(which_model=save_method,
               model_type=model_type,
               path_to_model_ovrd=ckp_path)
print(mgr.model.label)

# load data
mgr.load_data(mm.loaderRegister.IWSLT)
print(mgr.lparams)

# init record
true = []
pred = []
id2token = mgr.dataloader.id2token[iwslt.TAR]
max_len = mgr.model.hparams[HyperParamKey.MAX_LENGTH]

with torch.no_grad():
    for src, tgt, slen, _ in mgr.dataloader.loaders['dev']:
        for idx in range(len(src)):
            # get sample
            src_i = src[idx].unsqueeze(0)
            tgt_i = tgt[idx].unsqueeze(0)
            slen_i = slen[idx].unsqueeze(0)
            # tune to eval mode
            mgr.model.encoder.eval()
            mgr.model.encoder.eval()
            # encoding
            enc_out, hidden = mgr.model.encoder(src_i, slen_i)
            # Greedy search
            greedy = mgr.model.decoding(tgt_i, (enc_out, hidden), False, mode='translate')

            # my beam search
            # pad enc_out batch
            if enc_out.size(1) < 100:
                enc_out = torch.cat((enc_out, torch.zeros((1,
                                                           max_len - enc_out.size(1),
                                                           enc_out.size(2))).to(DEVICE)), dim=1)
            # decoding
            predicted = beam_search(None, hidden, enc_out, mgr.model.decoder, max_len,
                                    beam_width=beam_w, len_penalty_type=len_penalty)
            # convert to strings
            try:
                target = " ".join([id2token[e] for e in tgt_i.squeeze() if e != iwslt.PAD_IDX])
                print("***\n***\nTRUTH:  ", target)
                greedy_tran = " ".join([id2token[e] for e in greedy])
                print("GREEDY: ", greedy_tran)
                translated = " ".join([id2token[e] for e in predicted[1:]])
                print("BEAM:   ", translated)
                # print("Beam score: (A){}  Greedy score: {}".format(
                #     bleuScorer.bleu(target, translated, score_only=True),
                #     bleuScorer.bleu(target, greedy_tran, score_only=True)))
                true.append(target)
                pred.append(translated)
            except:
                print(str(predicted))

print(bleuScorer.bleu(true, [pred], score_only=False))
