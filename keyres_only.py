import os
import argparse
import re

import pandas as pd
import numpy as np
from generate_esm2_attns import load_model, embed_single_seq
from utils import read_fasta, read_fasta_folder, DiscardHead
from get_key_residues import extract_keyres_from_single_seq
from attn_to_dictionary import (dictionary_longwords_percentage, continuous_words_separation,
                                words_classify_and_reassign, dictionary_raw_counts_threshold)
from main import single_seq_workflow
import gc
import pickle
import requests



def main():
    parser = argparse.ArgumentParser()
    WORKDIR = os.path.abspath(os.path.join('.')) + '/'
    max_length = 1024
    legal_aa_list = set([i for i in 'AVLIMPFWGSTYCNQDEHKR'])
    print(WORKDIR)
    parser.add_argument(
        "-i", "--fasta_name", type=str, required=True,
        help="Path of fasta files")

    parser.add_argument(
        "-o", "--out_dir", type=str, default=os.path.join(WORKDIR, 'outputs/'),
        help="Path of output attention directory")

    parser.add_argument(
        "--ignored_head_file", type=str, default=os.path.join(WORKDIR, 'data/head_cls_dict.pkl'),
        help="Path of output attention directory")



    args = parser.parse_args()
    fasta_name = os.path.basename(args.fasta_name).split('.')[0]
    # Define the name of the dataset by fasta file or path name
    if args.fasta_name.endswith('.fasta'):
        seq_names, seqs = read_fasta(args.fasta_name)
        base_name = os.path.basename(args.fasta_name).replace('.fasta', '')
    elif re.fullmatch('[a-zA-Z0-9]+', args.fasta_name):
        response = requests.get(f'https://rest.uniprot.org/uniprotkb/{args.fasta_name}.fasta')
        if response.status_code == 200:
            seq_names = [args.fasta_name]
            seqs = [''.join(response.text.split('\n')[1:])]
            print(f'Sequence {args.fasta_name} download success!')
        else:
            seq_names, seqs = [], []

    elif os.path.basename(args.fasta_name) != '':
        seq_names, seqs = read_fasta(args.fasta_name)
        base_name = os.path.basename(args.fasta_name).split('.')[0]
    else:
        seq_names, seqs = read_fasta_folder(args.fasta_name)
        base_name = os.path.dirname(args.fasta_name).split('/')[-1]


    print(base_name, seq_names)
    output_path = args.out_dir
    os.makedirs(output_path, exist_ok=True)

    all_seq_dict = {k: v for k, v in zip(seq_names, seqs)
                    if len(v) <= max_length and set(np.unique([i for i in v])).issubset(legal_aa_list)}
    # Step 1: Run ESM2 to retrieve attention matrices
    model, alphabet, batch_converter, device = load_model()
    # Read head classification dict and determine which heads are to be ignored
    head_cls_dict = pd.read_pickle(args.ignored_head_file)
    n_sequences = len(list(all_seq_dict.keys()))
    list_df_keyres_all = []
    # list_df_wordlist_all = []
    output_keyres_filename = f'{fasta_name}_keyres.csv'
    last_checkpoint_path = None
    for i, (metadata, seq) in enumerate(all_seq_dict.items()):

        if '|' in metadata:
            name = metadata.split('|')[0]
        else:
            name = metadata
        df_keyres, _ = single_seq_workflow(name, seq, model, alphabet, head_cls_dict)
        list_df_keyres_all.append(df_keyres)
        if (i+1)%100 == 0:
            temp_filename = f'{fasta_name}_keyres_checkpoint{(i+1)//100}.csv'
            df_keyres_all_raw = pd.concat(list_df_keyres_all)
            if last_checkpoint_path is not None:
                os.remove(last_checkpoint_path)
            last_checkpoint_path = os.path.join(args.out_dir, temp_filename)
            df_keyres_all_raw.to_csv(last_checkpoint_path)

        print(f'Sequence {i}/{n_sequences}: {name} success!')
        del df_keyres
        gc.collect()

    df_keyres_all_raw = pd.concat(list_df_keyres_all)
    df_keyres_all_raw.to_csv(
        os.path.join(args.out_dir, output_keyres_filename))
    if last_checkpoint_path is not None:
        os.remove(last_checkpoint_path)
    return 0


if __name__=='__main__':
    main()