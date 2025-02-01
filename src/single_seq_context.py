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


def find_uniprot_context(uniprot_id, lookup_dict, prefix_path, pfam_list=None):
    #lut = pd.read_csv(list_path)
    if pfam_list is None:
        if uniprot_id in lookup_dict.keys():
            pfam_list = [i for i in lookup_dict[uniprot_id].split(' ') if i !='']
        else:
            return np.array([])
    #pfam_list = list(lut[lut['uniprot_acc']==uniprot_id]['pfamA_acc'])

    #CSV version (legacy)
    # path_list = [os.path.join(prefix_path, i, 'merge') for i in pfam_list]
    # path_list = [i for i in path_list if os.path.isdir(i)]
    # file_list = [[os.path.join(i,j) for j in os.listdir(i) if j.endswith('.csv')] for i in path_list]
    # NPY version
    file_list = [os.path.join(prefix_path, f'{i}.npy') for i in pfam_list]
    file_list = [i for i in file_list if os.path.isfile(i)]

    return np.array(file_list)

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
        "-s", "--self_count_threshold", type=int, default=1000,
        help="Self Count Threshold")

    parser.add_argument(
        "-e", "--pfam_word_num", type=int, default=200,
        help="Number of words in each Pfam dictionary (not implemented)")

    parser.add_argument(
        "--ignored_head_file", type=str, default=os.path.join(WORKDIR, 'data/head_cls_dict.pkl'),
        help="Path of output attention directory")
    parser.add_argument(
        "-r",  "--ur50", action="store_true",
        help="Use UniRef50 dictionary  (1M sequences) as general context"
    )
    parser.add_argument(
        "-x",  "--ur50_only", action="store_true",
        help="Use UniRef50 dictionary  (1M sequences) only as context"
    )
    parser.add_argument(
        "-l",  "--min_frags", type=int, default=1,
        help="Minimum number of continuous segments"
    )
    parser.add_argument(
        "-u",  "--max_frags", type=int, default=3,
        help="Maximum number of continuous segments"
    )
    parser.add_argument(
        "--lookup_dict_path", type=str, default=os.path.join(WORKDIR, 'data/pfam_lookup.pkl'),
        help="Path of Pfam lookup dict"
    )
    parser.add_argument(
        "--dict_path", type=str, default=os.path.join(WORKDIR, 'data/uniref50_dictionaries'),
        help="Path of Pfam dictionaries"
    )
    parser.add_argument(
        "--uniref50_path", type=str, default=os.path.join(WORKDIR, 'data/ur50_1m.npy'),
        help="Path of UniRef reference dictionary"
    )
    parser.add_argument(
        "-k", "--save_key_residues", action="store_true",
        help="Save key residues"
    )


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

    list_path = args.lookup_dict_path
    with open(list_path, 'rb') as f:
        f.seek(0)
        lookup_dict = pickle.load(f)

    ur50_path = args.uniref50_path

    all_seq_dict = {k: v for k, v in zip(seq_names, seqs)
                    if len(v) <= max_length and set(np.unique([i for i in v])).issubset(legal_aa_list)}
    # Step 1: Run ESM2 to retrieve attention matrices
    model, alphabet, batch_converter, device = load_model()
    # Read head classification dict and determine which heads are to be ignored
    head_cls_dict = pd.read_pickle(args.ignored_head_file)
    n_sequences = len(list(all_seq_dict.keys()))
    list_df_keyres_all = []
    # list_df_wordlist_all = []
    for i, (metadata, seq) in enumerate(all_seq_dict.items()):

        if '|' in metadata:
            name = metadata.split('|')[0]
            pfam_list = metadata.split('|')[1].split(',')
        else:
            name = metadata
            pfam_list = None
        df_keyres, df_wordlist = single_seq_workflow(name, seq, model, alphabet, head_cls_dict)
        list_df_keyres_all.append(df_keyres)
        #list_df_wordlist_all.append(df_wordlist)

        uniprot_ids = name.split('_')[0].split(',')
        desired_dict = df_wordlist[['word_type', 'chain']]

        if not args.ur50_only:
            context_file_list = np.concatenate([find_uniprot_context(ids, lookup_dict=lookup_dict,
                                    prefix_path=args.dict_path, pfam_list=pfam_list) for ids in uniprot_ids])
            context_file_list = np.unique(context_file_list)
            if args.ur50 or args.ur50_only:
                context_file_list = np.append(context_file_list, ur50_path)
        else:
            context_file_list = np.array([ur50_path])

        all_contexts = []
        dictionary_self, _ = dictionary_raw_counts_threshold(desired_dict, threshold=args.self_count_threshold)
        all_contexts.append(dictionary_self)

        for context_file in context_file_list:
            print(f'Context file:{context_file}')
            try:
                if context_file.endswith('npy'):
                    dictionary = np.load(context_file, allow_pickle=True)
                elif context_file.endswith('csv'):
                    table_csv = pd.read_csv(context_file)
                    dictionary = np.unique(table_csv[table_csv.iloc[:,0]=='words']['word_type'])
                    del(table_csv)
                elif context_file.endswith('tsv'):
                    table_csv = pd.read_csv(context_file, sep='\t')
                    dictionary = np.unique(table_csv[table_csv.iloc[:,0]=='words']['word_type'])
                    del(table_csv)
                else:
                    dictionary = np.array([])
                gc.collect()
                all_contexts.append(dictionary)
            except FileNotFoundError:
                print('File Not Found!', context_file)
                continue
        dictionary = np.unique(np.concatenate(all_contexts))
        desired_dict = words_classify_and_reassign(df_wordlist,
                                                   min_frags=args.min_frags, max_frags=args.max_frags)
        word_dataframe_bycount_split = desired_dict[desired_dict['word_type'].isin(dictionary)]
        if len(name) > 30:
            name = name[-30:]
        if args.ur50_only:
            output_filename = f'{name}_crosspfam_self_{args.self_count_threshold}_ur50_only.csv'
        elif args.ur50:
            output_filename = f'{name}_crosspfam_self_{args.self_count_threshold}_ur50.csv'
        else:
            output_filename = f'{name}_crosspfam_self_{args.self_count_threshold}.csv'
        word_dataframe_bycount_split.to_csv(
            os.path.join(args.out_dir, output_filename))
        print(f'Sequence {i}/{n_sequences}: {name} success!')
        del desired_dict, df_wordlist, df_keyres
        gc.collect()
    output_keyres_filename = f'{fasta_name}_keyres.csv'
    df_keyres_all_raw = pd.concat(list_df_keyres_all)
    df_keyres_all_raw.to_csv(
        os.path.join(args.out_dir, output_keyres_filename))

    return 0


if __name__=='__main__':
    main()


'''USAGE'''
'''
Use Pfam dictionary only for word search:
python single_seq_context.py -i ./dataset/example.fasta -o ./outputs/pfam

Use both Pfam dictionary + UniRef50 general dictionary 
python single_seq_context.py -i ./dataset/example.fasta -o ./outputs/pfam_ur50_both -r

Use UniRef50 dictionary only
python single_seq_context.py -i ./dataset/example.fasta -o ./outputs/pfam_ur50_both -x

output format:
a csv table of detected words for each input sequence, each row contains the metadata of a word, 
    including chain, head information,  word type, original sequence, position in the sequence and pagerank importance order

parameters:
-i : input filename
-o : output filename
-r : use pfam+uniref50 combined context
-x : use uniref50 context only
-l : min number of continuous segments in a word (default: 1)
-u : max number of continuous segments in a word (default: 3)
-k : save key residue information (as a separate file)
'''
