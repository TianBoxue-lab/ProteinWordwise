import pandas as pd
import os,re
import numpy as np
from utils import ResidueClassification
def extract_keyres_from_single_seq(seq,seq_name,attns,ss3,head_cls_dict,ratio_cutoff = 0.1, avg_mutual_cutoff = 1.):
    excluded_heads = np.ones((33, 20))
    if ss3:
        df = pd.DataFrame(columns=['chain', 'layer', 'head', 'word_type', 'pos', 'seq_restype', 'seq', 'head_type', 'ss3'])
    else:
        df = pd.DataFrame(columns=['chain', 'layer', 'head', 'word_type', 'pos', 'seq_restype', 'seq', 'head_type'])
    idx = 0
    seq_length = len(seq)
    max_cutoff = int(seq_length * ratio_cutoff)
    # define desired head type(s) here
    desired_headtypes = ['keyres']
    for head_type, head_names in head_cls_dict.items():
        if head_type in desired_headtypes:
            #print(head_type)
            for head_name in head_names:
                key_residue_layer, key_residue_head = get_layer_head(head_name)
                key_resi_mat = attns[key_residue_layer, key_residue_head]
                tmp = np.sum(key_resi_mat, 0)

                # Sort residues by attention scores
                sorted_indices = np.argsort(tmp)[::-1]
                sorted_data = tmp[sorted_indices]
                key_res_indexes_sorted = sorted_indices[np.where(sorted_data >= max_cutoff)[0]]

                # Do not sort residues by attention scores
                key_res_indexes = np.where(tmp >= max_cutoff)[0]
                avg_residual_contacts = np.sum(tmp[np.where(tmp < max_cutoff)[0]])/len(np.where(tmp < max_cutoff)[0])
                if avg_residual_contacts > 1.:
                    excluded_heads[key_residue_layer, key_residue_head] = 0

                if len(key_res_indexes) > 0:
                    keyres_number = len(key_res_indexes)
                    keyres_mutual = key_resi_mat[key_res_indexes, :][:, key_res_indexes]
                    rows, cols = np.diag_indices_from(keyres_mutual)
                    keyres_mutual[rows, cols] = 0
                    if (keyres_number == 1) or (np.sum(keyres_mutual) < avg_mutual_cutoff * (keyres_number-1)):
                        key_res = ''.join([seq[int(i)] for i in key_res_indexes])
                        key_res_sorted = ''.join([seq[int(i)] for i in key_res_indexes_sorted])
                        if ss3:
                            sub_ss3 = ''.join([ss3[i] for i in key_res_indexes])
                            df.loc[idx] = (
                            seq_name, key_residue_layer, key_residue_head, ResidueClassification(key_res_sorted), \
                            key_res_indexes, ResidueClassification(key_res), key_res, head_type, sub_ss3)
                        else:
                            df.loc[idx] = (
                            seq_name, key_residue_layer, key_residue_head, ResidueClassification(key_res_sorted), \
                            key_res_indexes, ResidueClassification(key_res), key_res, head_type)

                        idx += 1

    return df, excluded_heads


def extract_key_residues(seq_dict,attns_dict,ss3_filename,output_path, head_cls_dict,max_cutoff = 10):
    #head_cls_dict = pd.read_pickle('./scripts/head_cls_dict.pkl') # read head classification dict
    if ss3_filename:
        ss3_dict = pd.read_pickle(ss3_filename)
    all_df = []
    excluded_heads_dict = {}


    for name, seq in seq_dict.items():
        # attns = pd.read_pickle(os.path.join(attn_dir, f'{name}_all_heads.pkl'))  # read attention mats
        attns = attns_dict[name]    # retrieve attention of a sequence from a dict
        if ss3_filename:
            df, excluded_heads = extract_keyres_from_single_seq(seq,name,attns,
                                      ss3 = ss3_dict[name],
                                      head_cls_dict=head_cls_dict,
                                      max_cutoff=max_cutoff)
        else:
            df, excluded_heads = extract_keyres_from_single_seq(seq,name,attns,
                                      ss3=None,
                                      head_cls_dict=head_cls_dict,
                                      max_cutoff=max_cutoff)
        all_df.append(df)
        excluded_heads_dict[name] = excluded_heads
        print('Finished extracting key_residues: %s'%name)
    merged_df = pd.concat(all_df)
    merged_df.to_csv(output_path)
    return merged_df, excluded_heads_dict   # to be filled with head discard algorithm

def get_layer_head(name):
    # get layer and head index
    tmp = re.findall('L(.*?)H(.*)', name)
    layer,head = tmp[0]
    return int(layer),int(head)


