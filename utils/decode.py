import os
import pdb
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        ctc_vocab = self.vocab[1:] 
        self.ctc_decoder = build_ctcdecoder(ctc_vocab)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        if not probs:
            nn_output = nn_output.softmax(-1)

        nn_output = torch.cat([nn_output[:, :, 1:], nn_output[:, :, 0:1]], dim=-1)

        nn_output = nn_output.cpu()
        vid_lgt = vid_lgt.cpu()
        nn_output_np = nn_output.numpy()

        batch_size = nn_output_np.shape[0]
        ret_list = []

        for batch_idx in range(batch_size):
            T = int(vid_lgt[batch_idx])
            logits = nn_output_np[batch_idx, :T, :]
            decoded_text = self.ctc_decoder.decode(logits)
            index_seq = [self.char2idx[ch] for ch in decoded_text if ch in self.char2idx]

            if len(index_seq) != 0:
                index_tensor = torch.tensor(index_seq, dtype=torch.long)
                index_tensor = torch.stack([x[0] for x in groupby(index_tensor)])
            else:
                index_tensor = []

            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(index_tensor)
            ])

        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list