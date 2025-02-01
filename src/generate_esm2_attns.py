import gc

import torch,esm,os
import numpy as np
import pandas as pd
from utils import ReadFromTar
from itertools import islice


def chunks(data, SIZE=32):
   it = iter(data)
   for i in range(0, len(data), SIZE):
      yield {k:data[k] for k in islice(it, SIZE)}


def load_model(model_name='t33_650m', use_cuda=False):
    print('Start loading model')
    if model_name == 't48_15b':
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    elif model_name == 't33_650m':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == 't12_35m':
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()  # disables dropout for deterministic results
    device = torch.device('cpu')
    #if use_cuda and torch.cuda.is_available():
    #    device = torch.device('cuda:0')
    #else:
    #    device = torch.device('cpu')
    model.to(device)
    print('Model loaded')
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, device


def embed_single_seq(seq, model, alphabet, batch_converter, device,
               drop_rate=0.4, averaging_rate=0.1):
    data = [(1, seq)]
    tmp = get_attention(data, batch_converter, model, device, alphabet)
    attns = tmp[0, :, :, 1: len(seq) + 1, 1: len(seq) + 1]
    binary_mats = prep_all_attn(attns, average_rate=averaging_rate, dropout_rate=drop_rate) # binary_mats: (l, h, N, N)
    del attns, tmp, data
 #   torch.cuda.empty_cache()
    gc.collect()
    return binary_mats


def embed_seqs(seq_dict, model, alphabet, batch_converter, device,
               drop_rate=0.4, averaging_rate=0.1):
    binary_mats_batch = {}
    for batch_seq_dict in seq_dict:
        data = [(1, seq)]
        tmp = get_attention(data, batch_converter, model, device, alphabet)
        for i, (name, seq) in enumerate(batch_seq_dict.items()):
            attns = tmp[i, :, :, 1: len(seq) + 1, 1: len(seq) + 1]
            binary_mats = prep_all_attn(attns, average_rate=averaging_rate, dropout_rate=drop_rate) # binary_mats: (l, h, N, N)
            binary_mats_batch[name] = binary_mats #dict: {name:binary_mats (l, h, N, N)}
        torch.cuda.empty_cache()
        gc.collect()
        print('Finished attention extraction: %s'%name)
    return binary_mats_batch


def get_attention(data, batch_converter, model, device, alphabet):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    # Extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens.to(device), return_contacts=True)
    return results['attentions']

def dropout_mat_by_percent(mat, drop_rate):
    flattened_mat = np.sort(mat.flatten())
    total_elements = flattened_mat.size
    elements_to_keep = int(total_elements * (1 - drop_rate))
    threshold_value = flattened_mat[elements_to_keep]
    binary_mat = np.where(mat > threshold_value, 1, 0)
    return binary_mat


def dropout_mat_by_value(mat, average_cutoff, drop_rate=None):
    min_v, max_v = np.min(mat), np.max(mat)
    threshold_value = (min_v + max_v) * average_cutoff
    binary_mat = np.where(mat > threshold_value, 1, 0)
    if drop_rate and len(np.where(binary_mat == 1)[0]) > len(mat) * len(mat) * drop_rate:
        binary_mat = dropout_mat_by_percent(mat, drop_rate)
    return binary_mat


def prep_all_attn(attn, dropout_rate=None, average_rate=None):
    tmp = np.zeros((attn.shape))
    for l in range(attn.shape[0]):  # layer index
        for h in range(attn.shape[1]):  # head index
            mat = attn[l, h, :, :].cpu().numpy()
            binary_mat = dropout_mat_by_value(mat, average_rate, dropout_rate)
            tmp[l, h] = binary_mat
    return tmp
