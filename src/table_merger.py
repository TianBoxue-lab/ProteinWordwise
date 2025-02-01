import os
import argparse
import pandas as pd
import numpy as np
from attn_to_dictionary import (dictionary_longwords_percentage, dictionary_longwords_adjusted,
                                continuous_words_separation,
                                words_classify_and_reassign)
import gc


def list_keyres_files(prefix, suffix='_keyres.csv'):
    prefix_dir = os.path.dirname(prefix)
    prefix_base = os.path.basename(prefix)
    files = os.listdir(prefix_dir)
    files_prefix = [i for i in files if (i.startswith(prefix_base) and i.endswith(suffix))]
    full_filename_filtered = [os.path.join(prefix_dir, i) for i in files_prefix]
    return full_filename_filtered


def main():
    parser = argparse.ArgumentParser()
    WORKDIR = os.path.abspath(os.path.join('.')) + '/'

    parser.add_argument(
        "--prefix", type=str, required=True,
        help="Path of fasta files")
    parser.add_argument(
        "--dict_size", type=int, default=300,
        help="Size of high-frequency dictionary")
    parser.add_argument(
        "--out_dir", type=str, default=os.path.join(WORKDIR, 'outputs/'),
        help="Path of output attention directory")
    parser.add_argument(
        "--ext_dict", type=str, default='',
        help="External dictionary or segmentation table")
    parser.add_argument(
        "-b", "--use_both_dicts", action="store_true",
        help="Use external together with self dict")
    parser.add_argument(
        "-cs", "--threshold_cs", type=float, required=True,
        help="Threshold parameter for short continuous words")
    parser.add_argument(
        "-cl", "--threshold_cl", type=float, required=True,
        help="Threshold parameter for long continuous words")
    parser.add_argument(
        "-ds", "--threshold_ds", type=float, default=None,
        help="Threshold parameter for short discontinuous words")
    parser.add_argument(
        "-dl", "--threshold_dl", type=float, default=None,
        help="Threshold parameter for long discontinuous words")
    parser.add_argument(
        "--space_threshold", type=int, default=2,
        help="Maximum spaces in a continuous word segment")
    parser.add_argument(
        "-l",  "--min_cont_segments", type=int, default=None,
        help="Minimum number of continuous segments"
    )
    parser.add_argument(
        "-u",  "--max_cont_segments", type=int, default=None,
        help="Maximum number of continuous segments"
    )
    args = parser.parse_args()
    filename_prefix = args.prefix
    output_path = args.out_dir
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    keyres_raw_list = []
    dict_raw_wordtype = []
    dict_raw_chain = []
    keyres_filename_list = list_keyres_files(filename_prefix, '_keyres.csv')
    dict_filename_list = list_keyres_files(filename_prefix, '_dict_raw.csv')

    #def arguments
    arg_cs = args.threshold_cs
    arg_cl = args.threshold_cl
    arg_ds = args.threshold_ds
    arg_dl = args.threshold_dl
    if arg_ds is None:
        arg_ds = arg_cs
    if arg_dl is None:
        arg_dl = arg_cl
    param_name = f'var_{arg_cs}_{arg_cl}_{arg_ds}_{arg_dl}'
    percentage_list_cont = [arg_cs, arg_cs/2, arg_cl, arg_cl, arg_cl, arg_cl, arg_cl, arg_cl*2, arg_cl*2,
                            arg_cl*2, arg_cl*2, arg_cl*2, arg_cl*2, arg_cl*2, arg_cl*2, arg_cl*2]
    percentage_list_discont = [arg_ds, arg_ds/2, arg_dl, arg_dl, arg_dl, arg_dl, arg_dl, arg_dl*2, arg_dl*2,
                               arg_dl*2, arg_dl*2, arg_dl*2, arg_dl*2, arg_dl*2, arg_dl*2, arg_dl*2]

    # define length and fragment ranges
    min_frags = args.min_cont_segments
    max_frags = args.max_cont_segments
    if min_frags is None:
        min_frags = 1
    if max_frags is None:
        max_frags = 1e8
    length_range = ''
    if min_frags is not None:
        length_range = length_range + f'L{min_frags}'
    if max_frags is not None:
        length_range = length_range + f'U{max_frags}'
    if length_range == '':
        length_range = 'ALL'

    ######
    # Read all csv files in the format of <filename_prefix>_split<split_number>_dict_raw.csv and _keyres.csv
    # All split tables that was failed to generate was ignored
    ######
    for filename in keyres_filename_list:
        keyres_raw_read = pd.read_csv(filename, index_col=0,  lineterminator='\n', on_bad_lines='skip')
        keyres_raw_list.append(keyres_raw_read)
        del keyres_raw_read
    if len(keyres_filename_list)>0:
        df_keyres_all_raw = pd.concat(keyres_raw_list)
    else:
        df_keyres_all_raw = pd.DataFrame([])
    del keyres_raw_list
    gc.collect()
    ######
    # Read all clustering words by louvain community discovery
    #
    ######
    last_checkpoint_path = None
    if (args.ext_dict == '') or (args.use_both_dicts):
        for i, filename in enumerate(np.sort(dict_filename_list)):
            print(f'Reading {filename}')
            dict_raw_read = pd.read_csv(filename, index_col=0,  lineterminator='\n', on_bad_lines='skip')
            desired_dict = words_classify_and_reassign(dict_raw_read, min_frags=min_frags, max_frags=max_frags)
            word_raw_wordtype = list(desired_dict['word_type'])
            word_raw_chain = list(desired_dict['chain'])
            dict_raw_wordtype = dict_raw_wordtype+ word_raw_wordtype
            dict_raw_chain = dict_raw_chain + word_raw_chain
            print(f'Raw dictionary size: {len(word_raw_wordtype)}')
            del dict_raw_read
            gc.collect()
            # dict_raw_wordtype = list(np.concatenate(dict_raw_wordtype))
            # dict_raw_chain = list(np.concatenate(dict_raw_chain))
            desired_dict = pd.DataFrame(np.array([dict_raw_wordtype, dict_raw_chain]).T, columns=['word_type', 'chain']).drop_duplicates()
            desired_dict.to_csv(os.path.join(output_path, f'{os.path.basename(filename_prefix)}_checkpoint_{i}.csv'))
            last_checkpoint_path = os.path.join(output_path, f'{os.path.basename(filename_prefix)}_checkpoint_{i}.csv')
            if i>=1:
                os.remove(os.path.join(output_path, f'{os.path.basename(filename_prefix)}_checkpoint_{i-1}.csv'))



        desired_dict = pd.DataFrame(np.array([dict_raw_wordtype, dict_raw_chain]).T, columns=['word_type', 'chain'])
        ADJUSTED_TEST=False
        if ADJUSTED_TEST:
            res_prob_baseline =     {'@': 0.05176205826262859, 'C': 0.015274306019977143, 'H': 0.022329879118662035, 'M': 0.02208759968277109,
                                    'P': 0.05056184469422176, 'Y': 0.03068898564069333, 'a': 0.12107052738631748, 'b': 0.11377016818029986,
                                    'h': 0.21368956262504032, 'n': 0.08296851087198037, 'o': 0.1311366771601932, 's': 0.14465988035721483}

            dictionary_cont, _ = dictionary_longwords_adjusted(desired_dict, res_prob_baseline, dict_threshold=percentage_list_cont,
                                                                 min_frags=min_frags, max_frags=min(1, max_frags))
            dictionary_discont, _ = dictionary_longwords_adjusted(desired_dict,res_prob_baseline,
                                                                    dict_threshold=percentage_list_discont,
                                                                    min_frags=max(2, min_frags), max_frags=max_frags)
        else:
            dictionary_cont, _ = dictionary_longwords_percentage(desired_dict, dict_threshold=percentage_list_cont,
                                                                 min_frags=min_frags, max_frags=min(1, max_frags))
            dictionary_discont, _ = dictionary_longwords_percentage(desired_dict, dict_threshold=percentage_list_discont,
                                                                    min_frags=max(2, min_frags), max_frags=max_frags)

        dictionary_self = np.concatenate([dictionary_cont, dictionary_discont])

    if args.ext_dict == '':
        dictionary = dictionary_self
        del dictionary_self

    else:
        if args.ext_dict.endswith('npy'):
            dictionary_ext = np.load(args.ext_dict, allow_pickle=True)
        elif args.ext_dict.endswith('csv'):
            table_csv = pd.read_csv(args.ext_dict)
            dictionary_ext = np.unique(table_csv[table_csv.iloc[:,0]=='words']['word_type'])
        elif args.ext_dict.endswith('tsv'):
            table_csv = pd.read_csv(args.ext_dict, sep='\t')
            dictionary_ext = np.unique(table_csv[table_csv.iloc[:,0]=='words']['word_type'])
        if args.use_both_dicts:
            dictionary = np.unique(np.concatenate([dictionary_ext, dictionary_self]))
            del dictionary_ext, dictionary_self
        else:
            dictionary = dictionary_ext
            del dictionary_ext
    gc.collect()


    dict_selected_list = []

    for i, filename in enumerate(dict_filename_list):
        print(f'Analyzing #{i}: {filename}')
        dict_raw_read = pd.read_csv(filename, index_col=0,  lineterminator='\n', on_bad_lines='skip')
        desired_dict = words_classify_and_reassign(dict_raw_read, min_frags=min_frags, max_frags=max_frags)
        word_dataframe_bycount_split = desired_dict[desired_dict['word_type'].isin(dictionary)]
        del dict_raw_read
        del desired_dict
        dict_selected_list.append(word_dataframe_bycount_split)
        gc.collect()

    word_dataframe_bycount = pd.concat(dict_selected_list)
    df_combined_keyres_word = pd.concat([df_keyres_all_raw, word_dataframe_bycount], axis=0,
                                        keys=['keyres', 'words'], join='outer')

    df_combined_keyres_word.to_csv(os.path.join(output_path,
                                                f'{os.path.basename(filename_prefix)}_{length_range}_{param_name}.csv'))
    if last_checkpoint_path is not None:
        os.remove(last_checkpoint_path)
    return dictionary


if __name__ == '__main__':
    main()

'''
A script for merging all word tables of a split dataset (for parallel computing) produced by main.py
USAGE： python table_merger.py --prefix ./<prefix> --out_dir <output_path>
'''