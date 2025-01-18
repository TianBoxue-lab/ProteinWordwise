import os, argparse
import numpy as np
import pandas as pd
from generate_esm2_attns import load_model, embed_single_seq
from utils import read_fasta, read_fasta_folder, DiscardHead
from get_key_residues import extract_keyres_from_single_seq
from attn_to_dictionary import single_seq_community_detection, single_seq_wordlist
import torch


def single_seq_workflow(name, seq, model, alphabet, head_cls_dict, clf_mode=3):
    torch.set_num_threads(1)
    batch_converter = alphabet.get_batch_converter()
    device = torch.device('cpu')
    attentions = embed_single_seq(seq, model, alphabet, batch_converter, device)
    del model, alphabet, batch_converter, device

    print('Finished attention extraction: %s' % name)
    ss3 = None
    df_keyres, excluded_heads = extract_keyres_from_single_seq(seq, name, attentions, ss3, head_cls_dict,
                                                               ratio_cutoff=0.1, avg_mutual_cutoff=2.)
    manual_discard_heads = DiscardHead(head_cls_dict)
    ignored_heads = np.logical_and(manual_discard_heads, excluded_heads)
    branches_louvain = single_seq_community_detection(attentions, ignored_heads)
    df_wordlist = single_seq_wordlist(name, seq, attentions, branches_louvain, ignored_heads, clf_mode=clf_mode)
    print('Finished wordlist generation: %s' % name)
    return df_keyres, df_wordlist


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
        "--mode", type=str, required=True,
        help="Specific mode to run: single or batch")

    parser.add_argument(
        "--ss3_file", type=str,
        help="Path of ss3 file, containing secondary structures of input sequences")

    parser.add_argument(
        "--dict_dir", type=str, default=os.path.join(WORKDIR, 'tmp/'),
        help="Path of predefined dictionary (only for single sequence mode)")

    parser.add_argument(
        "-o", "--out_dir", type=str, default=os.path.join(WORKDIR, 'outputs/'),
        help="Path of output attention directory")

    parser.add_argument(
        "--ignored_head_file", type=str, default=os.path.join(WORKDIR, 'data/head_cls_dict.pkl'),
        help="Path of output attention directory")
    parser.add_argument(
        "-m", "--max_block_size", type=int, default=1e7,
        help="Path of output attention directory")
    parser.add_argument(
        "--clf_mode", type=int, default=3,
        help="Residue type classification mode")


    args = parser.parse_args()
    # Define the name of the dataset by fasta file or path name
    if args.fasta_name.endswith('.fasta'):
        seq_names, seqs = read_fasta(args.fasta_name)
        base_name = os.path.basename(args.fasta_name).replace('.fasta', '')
    elif os.path.basename(args.fasta_name) != '':
        seq_names, seqs = read_fasta(args.fasta_name)
        base_name = os.path.basename(args.fasta_name).split('.')[0]
    else:
        seq_names, seqs = read_fasta_folder(args.fasta_name)
        base_name = os.path.dirname(args.fasta_name).split('/')[-1]
    print(base_name, seq_names)

    # os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # single for transformation; batch for fitting
    if args.mode == "single":
        print('Input fasta name: {}'.format(args.fasta_name))
        print(f'Input sequence: {seqs[0]}\nSequence length: {len(seqs[0])}')
        #if len(seqs) != 1:
            #raise ValueError('Single mode not supported')

    else:
        print('Input fasta name: {}'.format(args.fasta_name))
        print(f'Containing %s sequences' % (len(seqs)))
        #if len(seqs) <= 1:
            #pass
            #raise ValueError('Batch mode not supported')
    all_seq_dict = {k: v for k, v in zip(seq_names, seqs)
                    if len(v) <= max_length and set(np.unique([i for i in v])[0]).issubset(legal_aa_list)}


    print(all_seq_dict.keys())
    print(args.ss3_file)
    output_path = args.out_dir
    # Step 1: Run ESM2 to retrieve attention matrices
    model, alphabet, batch_converter, device = load_model()
    # Read head classification dict and determine which heads are to be ignored
    head_cls_dict = pd.read_pickle(args.ignored_head_file)

    list_df_keyres_all = []
    list_df_wordlist_all = []
    max_block_size = args.max_block_size
    block_num = 1
    wordlist_counter = 0

    for i, (name, seq) in enumerate(all_seq_dict.items()):
        df_keyres, df_wordlist = single_seq_workflow(name, seq, model, alphabet, head_cls_dict, clf_mode=args.clf_mode)
        list_df_keyres_all.append(df_keyres)
        list_df_wordlist_all.append(df_wordlist)
        wordlist_counter += df_wordlist.shape[0]
        if wordlist_counter >= max_block_size:
            df_wordlist_all_raw = pd.concat(list_df_wordlist_all)
            df_wordlist_all_raw.to_csv(os.path.join(output_path, f'{base_name}_block{block_num}_dict_raw.csv'))
            del list_df_wordlist_all, df_wordlist_all_raw
            block_num += 1
    df_keyres_all_raw = pd.concat(list_df_keyres_all)
    df_keyres_all_raw.to_csv(os.path.join(output_path, f'{base_name}_keyres.csv'))
    df_wordlist_all_raw = pd.concat(list_df_wordlist_all)
    df_wordlist_all_raw.to_csv(os.path.join(output_path, f'{base_name}_block{block_num}_dict_raw.csv'))
    '''
    if args.mode == 'single':
        dictionary = np.load(args.dict_dit, allow_pickle=True)
        word_dataframe_bycount = df_wordlist_all_raw.loc[df_wordlist_all_raw['word_type'].isin(dictionary)]
    else:
        dictionary, word_dataframe_bycount = dictionary_longwords_percentage(df_wordlist_all_raw, dict_threshold=percentage_list)
        np.save(os.path.join(output_path, f'{base_name}_dictionary_rawcount.npy'), dictionary)

    # Save normalized dictionary and sequence segmentation table
    word_dataframe_bycount.to_csv(os.path.join(output_path, f'{base_name}_segment_table_rawcount.csv'))
    '''
    '''
    df_combined_keyres_word = pd.concat([df_keyres_all_raw, word_dataframe_bycount], axis=0,
                                        keys=['keyres', 'words'], join='outer')
    df_combined_keyres_word.to_csv(os.path.join(output_path, f'{base_name}_combined_keyres_and_words.csv'))
    '''
    '''
    # Step 4: Create word dictionary using normalized count
    res_prob_baseline = restype_freq_baseline(all_seq_dict)
    dictionary_normalized, word_dataframe_normalized = dictionary_normalized_counts(df_wordlist_all_raw, res_prob_baseline) 
    # Save normalized dictionary and sequence segmentation table
    np.save(os.path.join(output_path, f'{base_name}_dictionary_normalized.npy'), dictionary_normalized)
    word_dataframe_normalized.to_csv(os.path.join(output_path, f'{base_name}_segment_table_normalized.csv'))
    '''
    return df_keyres_all_raw, df_wordlist_all_raw


if __name__ == "__main__":
    main()



'''  
============================Requirements===============================================
conda create --name pt3.9 python=3.9
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (change cuda version)
Biopython==1.79
pandas==1.3.5
fair-esm==2.0
networkx==3.2.1
python==3.9.19
scipy==1.11.4

'''

'''
USAGE
python main.py -i <fasta_name> -o <output_name>

output format:
A single csv table or multiple tables (if row number exceeds max_row parameter) of all words and key residues detected by ESM2 in each input sequence, 
    the first column is "keyres" or "word", indicating the type of data in the row
    each "word" row contains the metadata of a word, 
    including chain, head information,  word type, original sequence, position in the sequence and pagerank importance order
    each "keyres" contains the data of all key residues determined by an attention head. 
    
parameters:
-i : input filename
-o : output filename
-m : max rows in one output table (default: 10000000). Redce this parameter to avoid memory overflow.
'''