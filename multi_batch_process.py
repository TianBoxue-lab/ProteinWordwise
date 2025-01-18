import os, argparse
from utils import read_fasta, read_fasta_folder
import numpy as np

def split_dict(dictionary, n_split):
    sub_dicts = []
    keys = list(dictionary.keys())
    total_keys = len(keys)
    split_size = total_keys // n_split+1
    for i in range(0, total_keys, split_size):
        sub_dict = {k : dictionary[k] for k in keys[i:i+split_size]}
        sub_dicts.append(sub_dict)

    return sub_dicts


def split_dict_by_value_length(dictionary, n_split):
    """
    Split a dictionary into n sub-dictionaries such that the total lengths
    of string values in each sub-dict are as balanced as possible.

    :param data_dict: Dictionary {key: string_value}
    :param n: Number of sub-dictionaries (bins) to create
    :return: A list of n sub-dictionaries
    """

    # 1. Convert the dictionary into a list of (key, value) pairs
    items = list(dictionary.items())
    # 2. Sort items by descending length of their string value
    items.sort(key=lambda x: len(x[1]), reverse=True)
    # 3. Initialize n sub-dictionaries and track their total string lengths
    subdicts = [{} for _ in range(n_split)]
    lengths = [0] * n_split  # keep track of cumulative string length in each sub-dict
    # 4. Distribute the items using a greedy "smallest bin" approach
    for key, val in items:
        min_index = lengths.index(min(lengths))
        subdicts[min_index][key] = val
        lengths[min_index] += len(val)
    return subdicts

def write_fasta(path, seq_dict):
    keys = seq_dict.keys()
    with open(path, 'w') as f:
        for name in keys:
            f.writelines(['>'+name+'\n', str(seq_dict[name])+'\n'])
    return path


def main():
    parser = argparse.ArgumentParser()
    WORKDIR = os.path.abspath(os.path.join('.')) + '/'
    max_length = 1024
    legal_aa_list = set(['A', 'V', 'L', 'I', 'M', 'P', 'F', 'W', 'G', 'S', 'T', 'Y', 'C', 'N', 'Q', 'D', 'E', 'H', 'K', 'R'])
    print(WORKDIR)

    parser.add_argument(
        "--fasta_name", type=str, required=True,
        help="Path of fasta files")
    parser.add_argument(
        "--n_cores", type=int, required=True,
        help="Number of cores to split the task")
    parser.add_argument(
        "--out_dir", type=str, default=os.path.join(WORKDIR, 'outputs/'),
        help="Path of output attention directory")

    args = parser.parse_args()
    n = args.n_cores
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
    all_seq_dict = {k: v for k, v in zip(seq_names, seqs) if (len(v) <= max_length) and (set(np.unique([i for i in v])).issubset(legal_aa_list))}
    sub_seq_dicts = split_dict_by_value_length(all_seq_dict, args.n_cores)
    for i, subdict in enumerate(sub_seq_dicts):
        write_fasta(os.path.join(args.out_dir, f'{base_name}_split{i+1}.fasta'), subdict)

    return None


if __name__ == "__main__":
    main()
