import gc
import os, sys
import numpy as np
import pandas as pd
import math
from scipy.sparse.csgraph import connected_components
import networkx as nx
from utils import ReadFromTar, ResidueClassification, TimeOut
import re

# final size of final dictionary (both by raw and normalized counts)
DICT_SIZE = 100
# define the raw count percentile threshold for a word to be included in the normalized count dictionary
#  (larger value will result in higher possibility to include rare words)
RAWCOUNT_FILTER_PERCENTAGE = 0.05

def ReadFastaFolder(path):
    def _readfastafile(fasta_file):
        with open(fasta_file, 'r') as f:
            seq = f.readlines()[-1].strip()
        return seq
    file_list = [i for i in os.listdir(path) if i[-6:]=='.fasta']
    return {i[:-6]: _readfastafile(os.path.join(path, i)) for i in file_list}


def LouvainTimeLimit(adj_matrix):
    @TimeOut(5)
    def limited_louvain(graph, seed=42):
        return nx.algorithms.community.louvain_communities(graph, seed=seed)
    G = nx.DiGraph()
    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)
    seed = 1
    while seed <= 3:
        try:
            partition = limited_louvain(G, seed=seed)
            return partition
        except TimeoutError:
            seed += 1
        except ZeroDivisionError:
            break
    partition = []
    # partition, _ = louvain_method(adj_matrix)
    return partition


def DensityFilter(matrix, comm, threshold=0.01):
    density_cal = matrix[list(comm),:][:, list(comm)]
    density_cal[np.arange(len(comm)), np.arange(len(comm))]=0
    return np.mean(density_cal) > threshold


def ConnectivityFilter(matrix, comm):
    connectivity = connected_components(matrix[list(comm), :][:, list(comm)])[0]==1
    return connectivity


def SortByPageRank(graph_matrix):
    num_nodes = graph_matrix.shape[0]
    G = nx.DiGraph()
    rows, cols = np.where(graph_matrix==1)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    pagerank_dict = nx.pagerank(G)
    for i in range(num_nodes):
        if i not in pagerank_dict.keys():
            pagerank_dict[i] = 0
    pagerank_values = [pagerank_dict[i] for i in range(num_nodes)]
    return np.argsort(pagerank_values)[::-1]


def UselessHeadParse(head_list):
    heads_disc = np.zeros((33,20))
    useless_heads = [[int(i) for i in head[1:].split('H')] for head in head_list]
    for i,j in useless_heads:
        heads_disc[i,j] = 1
    return heads_disc


def single_seq_community_detection(attentions, chain_ignored_heads):
    branches_louvain = np.zeros((attentions.shape[0], attentions.shape[1]), dtype=object)
    for layer in range(attentions.shape[0]):
        for head in range(attentions.shape[1]):
            if chain_ignored_heads[layer][head]:
                continue
            comm = LouvainTimeLimit(attentions[layer][head])
            comm = [i for i in comm if len(i) > 1]
            comm = [i for i in comm if DensityFilter(attentions[layer][head], i)]
            comm = [i for i in comm if ConnectivityFilter(attentions[layer][head], i)]
            branches_louvain[layer][head] = comm
    return branches_louvain


def single_seq_wordlist(chain_name, sequence, attentions, branches_louvain, chain_ignored_heads, clf_mode=3):
    words_full_list = []
    sequence_array = np.array([i for i in sequence])
    sequence_simplified = np.array([i for i in ResidueClassification(sequence, mode=clf_mode)])
    for layer in range(attentions.shape[0]):
        for head in range(attentions.shape[1]):
            if chain_ignored_heads[layer][head]:
                continue
            for branch in branches_louvain[layer][head]:
                node_num = len(branch)
                if node_num >= 5 and node_num <= 25:
                    np_branch = np.array(list(branch))
                    np_branch_sorted = np.sort(np_branch)

                    topo_branch = attentions[layer, head][np_branch_sorted][:, np_branch_sorted]
                    word_connections = np.sum(topo_branch)
                    word_pagerank = SortByPageRank(topo_branch)

                    word_seq_rescl = sequence_simplified[np_branch_sorted]
                    word_seq = sequence_array[np_branch_sorted]
                    #word_type = ''.join(np.sort([i for i in word_seq_rescl]))

                    word_type_sorted_by_pos = ''.join(np.array([i for i in word_seq_rescl]))
                    word_type = word_type_sorted_by_pos
                    word_seq_sorted_by_pos = ''.join(np.array([i for i in word_seq]))
                    word_type_sorted_pagerank = ''.join(np.array([i for i in word_seq_rescl])[word_pagerank])
                    word_seq_sorted_pagerank = ''.join(np.array([i for i in word_seq])[word_pagerank])
                    words_full_list.append([chain_name, layer, head, word_type, \
                                            np_branch_sorted, word_type_sorted_by_pos, word_seq_sorted_by_pos, \
                                            word_connections, word_pagerank,
                                            word_type_sorted_pagerank, word_seq_sorted_pagerank])
    if len(words_full_list)>0:
        word_dataframe = pd.DataFrame(words_full_list)
        word_dataframe.columns = ['chain', 'layer', 'head', 'word_type', 'pos', 'seq_restype', 'seq', 'num_edges',
                                  'pagerank', 'seq_restype_by_pagerank', 'seq_by_pagerank']
    else:
        word_dataframe = pd.DataFrame(columns=['chain', 'layer', 'head', 'word_type', 'pos', 'seq_restype', 'seq', 'num_edges',
                                  'pagerank', 'seq_restype_by_pagerank', 'seq_by_pagerank'])
    return word_dataframe


def dictionary_raw_counts(word_dataframe, dict_size=100):
    sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    dictionary = sequence_compositions[0][np.argsort(sequence_compositions[1])[::-1][:dict_size]]

    word_dataframe_bycount = word_dataframe.loc[word_dataframe['word_type'].isin(dictionary)]
    return dictionary, word_dataframe_bycount


def dictionary_counts_and_longwords(word_dataframe, dict_size=100):
    sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    dictionary_freq1 = sequence_compositions[0][np.argsort(sequence_compositions[1])[::-1][:dict_size]]
    dictionary_freq2 = [i for i in sequence_compositions[0][np.argsort(sequence_compositions[1])[::-1][:1000]] if len(i)>8]

    dictionary_len = [i for i in sequence_compositions[0] if len(i) > 12]
    dictionary = np.concatenate([dictionary_freq1, dictionary_freq2, dictionary_len])

    word_dataframe_bycount = word_dataframe.loc[word_dataframe['word_type'].isin(dictionary)]
    return dictionary, word_dataframe_bycount


def dictionary_raw_counts_threshold(word_dataframe, threshold=2):
    sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    dictionary = sequence_compositions[0][sequence_compositions[1] >= threshold]
    word_dataframe_bycount = word_dataframe.loc[word_dataframe['word_type'].isin(dictionary)]
    return dictionary, word_dataframe_bycount


def dictionary_longwords_percentage(word_dataframe, dict_threshold, min_frags=None, max_frags=None, return_df=False):
    if type(dict_threshold) is float:
        dict_threshold = np.ones(16) * dict_threshold
    # sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    # Count by raw words number
    '''
    chain_cts = word_dataframe.groupby('word_type')['word_type'].count()
    sequence_compositions = (np.array(chain_cts.index), np.array(chain_cts))
    '''
    # Count by words occurrence in different sequences
    chain_cts = word_dataframe.groupby(['word_type'])['chain'].unique()
    sequence_compositions = (np.array(chain_cts.index), np.array([len(i) for i in list(chain_cts)]))

    dictionary_lists = []
    discont_part_stat = np.array([i.count('_') for i in sequence_compositions[0]])
    if min_frags is None:
        min_frags = 1
    if max_frags is None:
        max_frags = 1e8
    #print(discont_part_stat[:100])
    selected_words = np.logical_and(discont_part_stat >= min_frags-1, discont_part_stat <= max_frags-1)
    sequence_compositions = (sequence_compositions[0][selected_words], sequence_compositions[1][selected_words])
    len_stat = np.array([len(i.replace('_', '')) for i in sequence_compositions[0]])

    for i, length in enumerate(range(5,21)):
        seq_len = sequence_compositions[0][len_stat == length]
        count_len = sequence_compositions[1][len_stat == length]
        num_unique_len = len(seq_len)
        #print(length, num_unique_len)
        if dict_threshold[0] <1:
            print(num_unique_len, dict_threshold[i])
            num_select_len = math.floor(num_unique_len * dict_threshold[i])
            dictionary_top = seq_len[np.argsort(count_len)[::-1][:num_select_len]]
            selected_number = np.sum(count_len[np.argsort(count_len)[::-1][:num_select_len]])
        if dict_threshold[0] >= 1:
            selected_els = np.where(count_len >= dict_threshold[0])[0]
            # print(selected_els)
            num_select_len = len(selected_els)
            dictionary_top = seq_len[selected_els]
            selected_number = np.sum(count_len[selected_els])

        print(f'Length {length}, dict size {num_select_len}, selected segments {selected_number}')
        dictionary_lists.append(dictionary_top)
        del seq_len, count_len, dictionary_top
        gc.collect()



    dictionary = np.concatenate(dictionary_lists)
    if return_df:
        word_dataframe_bycount = word_dataframe.loc[word_dataframe['word_type'].isin(dictionary)]
        return dictionary, word_dataframe_bycount
    else:
        return dictionary, None


def continuous_words_separation(word_dataframe, space_threshold=0):
    word_dataframe_filtered = word_dataframe.dropna(axis=0)
    word_df_pos = word_dataframe_filtered['pos']
    pos_int_list = [[int(j) for j in re.findall(r'\d+', i)] for i in word_df_pos]
    pos_int_list_filtered = [i for i in pos_int_list if len(i) > 0]
    word_dataframe_filtered = word_dataframe_filtered.iloc[np.where(np.array([len(i) for i in pos_int_list])>0)]
    pos_space_list = np.array([i[-1] - i[0] + 1 - len(i) for i in pos_int_list_filtered])
    return word_dataframe_filtered.iloc[pos_space_list <= space_threshold], word_dataframe_filtered.iloc[pos_space_list > space_threshold]


def words_classify_and_reassign(word_dataframe, space_threshold=0, min_frags=None, max_frags=None):
    def continuous_parts(pos_list, space_threshold):
        pos_list_sorted = np.sort(pos_list)
        pos_list_inc = pos_list_sorted[1:] - pos_list_sorted[:-1]

        if len(np.where(pos_list_inc > space_threshold+1)[0]) > 0:
            break_ps = (np.where(pos_list_inc > space_threshold + 1)[0] + 1)
            num_splits = len(break_ps)+1
            split_pos_list = []
            split_pos_list.append(pos_list_sorted[:break_ps[0]])
            for i in range(num_splits-2):
                split_pos_list.append(pos_list_sorted[break_ps[i]:break_ps[i+1]])
            split_pos_list.append(pos_list_sorted[break_ps[-1]:])
        else:
            num_splits=1
            split_pos_list = [np.sort(pos_list)]
        return num_splits, split_pos_list

    def reassign(word_dataframe, pos_int_list, space_threshold):
        #word_dataframe = word_dataframe_input.copy()
        cont_res_num = []
        word_type = []
        seq_degenerated =[]
        if 'seq_degenerated' in word_dataframe.columns:
            restype_colname = 'seq_degenerated'
        else:
            restype_colname = 'seq_restype'
        for i in range(word_dataframe.shape[0]):
            pos_list = pos_int_list[i]
            data_row = word_dataframe.iloc[i]
            seq_ident = np.array([k for k in data_row[restype_colname]])
            num_splits, split_pos_list = continuous_parts(pos_list, space_threshold)
            cont_res_num.append(num_splits)
            cursor = 0
            part_types_list = []
            part_seqdegs_list = []
            for part in split_pos_list:
                n_res_split = len(part)
                part_type = ''.join(np.sort(seq_ident[cursor:cursor+n_res_split]))
                part_types_list.append(part_type)
                part_seqdeg = ''.join(seq_ident[cursor:cursor+n_res_split])
                part_seqdegs_list.append(part_seqdeg)
                cursor += n_res_split
            part_types_list.sort()
            part_types_list.sort(key=len, reverse=True)
            word_type.append('_'.join(part_types_list))
            seq_degenerated.append('_'.join(part_seqdegs_list))
        word_dataframe['continuous_part_num'] = np.array(cont_res_num, dtype=int)
        # word_dataframe['word_type'] = word_type
        word_dataframe['word_type'] = seq_degenerated
        word_dataframe['seq_degenerated'] = seq_degenerated
        del word_type
        del seq_degenerated
        del cont_res_num
        gc.collect()
        if min_frags is not None:
            word_dataframe = word_dataframe[word_dataframe['continuous_part_num'] >= min_frags]
        if max_frags is not None:
            word_dataframe = word_dataframe[word_dataframe['continuous_part_num'] <= max_frags]
        return word_dataframe

    word_dataframe_filtered = word_dataframe.dropna(axis=0)
    word_df_pos = word_dataframe_filtered['pos']
    if isinstance(word_df_pos.iloc[0], str):
        pos_int_list = [[int(j) for j in re.findall(r'\d+', i)] for i in word_df_pos]
    else:
        pos_int_list = list(word_df_pos)
    pos_int_list_filtered = [i for i in pos_int_list if len(i) > 0]
    word_dataframe_filtered = word_dataframe_filtered.iloc[np.where(np.array([len(i) for i in pos_int_list])>0)]
    reassigned_word_dataframe = reassign(word_dataframe_filtered, pos_int_list_filtered, space_threshold)
    return reassigned_word_dataframe
def restype_freq_baseline(seq_dict_all):
    # Calculate frequency of each degenerated residue type
    all_types = np.array([]).astype(str)
    for sequence in seq_dict_all.values():
        sequence_simplified = np.array([i for i in ResidueClassification(sequence, 3)])
        all_types = np.concatenate((all_types, sequence_simplified))
    res_prob_baseline = np.unique(all_types, return_counts=True)[1] / np.sum(
        np.unique(all_types, return_counts=True)[1])
    res_prob_baseline = dict(zip(np.unique(all_types, return_counts=True)[0], res_prob_baseline))
    return res_prob_baseline


def dictionary_longwords_adjusted(word_dataframe, res_prob_baseline, dict_threshold, min_frags=None, max_frags=None, return_df=False):

    #dict_threshold: dict of percentages of each restype, format{type_i: percentage_i}
    if type(dict_threshold) is float:
        dict_threshold = np.ones(16) * dict_threshold
    # sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    # Calculate normalized count for every unique words
    wordfreq_adjusted = []
    for i, word in enumerate(sequence_compositions[0]):
        word_counts = sequence_compositions[1][i]
        wordfreq_adjusted.append(word_counts / np.prod([res_prob_baseline[i] * len(res_prob_baseline)
                                                        for i in word.replace('_', '')]))
    wordfreq_adjusted = np.array(wordfreq_adjusted)
    chain_cts = word_dataframe.groupby(['word_type'])['chain'].unique()
    sequence_compositions = (np.array(chain_cts.index), np.array([len(i) for i in list(chain_cts)]))
    dictionary_lists = []
    discont_part_stat = np.array([i.count('_') for i in sequence_compositions[0]])
    if min_frags is None:
        min_frags = 1
    if max_frags is None:
        max_frags = 1e8
    #print(discont_part_stat[:100])
    selected_words = np.logical_and(discont_part_stat >= min_frags-1, discont_part_stat <= max_frags-1)
    sequence_compositions = (sequence_compositions[0][selected_words], sequence_compositions[1][selected_words])
    wordfreq_adjusted  = wordfreq_adjusted[selected_words]
    len_stat = np.array([len(i.replace('_', '')) for i in sequence_compositions[0]])

    for i, length in enumerate(range(5,21)):
        seq_len = sequence_compositions[0][len_stat == length]
        # set raw count to adjusted word frequency here
        #count_len = sequence_compositions[1][len_stat == length]
        count_len = wordfreq_adjusted[len_stat == length]
        num_unique_len = len(seq_len)
        #print(length, num_unique_len)
        num_select_len = math.floor(num_unique_len * dict_threshold[i])
        dictionary_top = seq_len[np.argsort(count_len)[::-1][:num_select_len]]
        selected_number = np.sum(count_len[np.argsort(count_len)[::-1][:num_select_len]])
        print(f'Length {length}, dict size {num_select_len}, selected segments {selected_number}')
        dictionary_lists.append(dictionary_top)
        del seq_len, count_len, dictionary_top
        gc.collect()

    dictionary = np.concatenate(dictionary_lists)
    if return_df:
        word_dataframe_bycount = word_dataframe.loc[word_dataframe['word_type'].isin(dictionary)]
        return dictionary, word_dataframe_bycount
    else:
        return dictionary, None


def dictionary_normalized_counts(word_dataframe, res_prob_baseline, rawcount_filter=0.05, dict_size=100):
    sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    # Calculate normalized count for every unique words
    wordfreq_compensated = []
    for i, word in enumerate(sequence_compositions[0]):
        word_counts = sequence_compositions[1][i]
        wordfreq_compensated.append(word_counts / np.prod([res_prob_baseline[i] * len(res_prob_baseline) for i in word]))

    # Create a DataFrame of all words' raw and normalized count
    wordfreq_compensated = np.array(wordfreq_compensated)
    dictionary_dataframe = pd.DataFrame([sequence_compositions[0], sequence_compositions[1], wordfreq_compensated]).T
    dictionary_dataframe.columns = ['sequence', 'raw_count', 'normalized']
    dictionary_dataframe = dictionary_dataframe.sort_values(by=['raw_count', 'normalized'])
    dictionary_dataframe = dictionary_dataframe.iloc[-int(dictionary_dataframe.shape[0] * rawcount_filter):]
    common_compositions_corrected = list(dictionary_dataframe.sort_values(by='normalized').iloc[-dict_size:]['sequence'])
    word_dataframe_bycount = word_dataframe.loc[word_dataframe['word_type'].isin(common_compositions_corrected)]
    return common_compositions_corrected, word_dataframe_bycount


def extract_words(seq_dict_all, basename, attns_dict, output_path, ignored_heads):
    # run script command format: python community_to_dictionary.py <fasta_path> <embedding_path> <output_filename>
    '''file format:
    sequence: <fasta_path>/<chain1>.fasta, <chain2>.fasta ...
    attention: <embedding_path>/<chain1>_all.tar.gz , <chain2>_all.tar.gz ...
    '''
    # Step 0: Read files and determine chain names
    # attention_map_file_names = [i for i in os.listdir(embedding_path) if i.endswith('all_heads.pkl')]
    # chain_names = [i.split('_all_heads')[0] for i in attention_map_file_names if i.split('_all_heads')[0] in seq_dict_all.keys()]
    chain_names = list(seq_dict_all.keys())
    words_full_list = []

    # Step 1: Community discovery
    for cnum, chain_name in enumerate(chain_names):
        print('Start to extract words :', chain_name)
        sequence = seq_dict_all[chain_name]
        attentions = attns_dict[chain_name]  # pd.read_pickle(embedding_path+'/'+chain_name+'_all_heads.pkl')
        chain_ignored_heads = ignored_heads[chain_name]
        #   Louvain: branches_louvain
        branches_louvain = single_seq_community_detection(attentions, chain_ignored_heads)

        word_dataframe_single = single_seq_wordlist(chain_name, sequence, attentions, branches_louvain, ignored_heads)
        words_full_list.append(word_dataframe_single)

    # Step 2: Create table of communities and their metadata
    word_dataframe = pd.concat(words_full_list)
    word_dataframe.to_csv(os.path.join(output_path, f'{basename}_dict_raw.csv'))

    # Step 3: Create word dictionary and segment table
    dictionary, word_dataframe_bycount = dictionary_raw_counts(word_dataframe)

    # Save normalized dictionary and sequence segmentation table
    np.save(os.path.join(output_path, f'{basename}_dictionary_rawcount.npy'), dictionary)
    word_dataframe_bycount.to_csv(os.path.join(output_path, f'{basename}_segment_table_rawcount.csv'))


    # Step 4: Create word dictionary using normalized count
    res_prob_baseline = restype_freq_baseline(seq_dict_all)
    dictionary_normalized, word_dataframe_normalized = dictionary_normalized_counts(word_dataframe, res_prob_baseline)

    # Save normalized dictionary and sequence segmentation table
    np.save(os.path.join(output_path, f'{basename}_dictionary_normalized.npy'), dictionary_normalized)
    word_dataframe_normalized.to_csv(os.path.join(output_path, f'{basename}_segment_table_normalized.csv'))

    return word_dataframe_bycount


#Test
if __name__=='__main__':
    #rawfile = pd.read_csv('../output/nano_dict_raw.csv', index_col=0)
    #processed_df = words_classify_and_reassign(rawfile, min_frags=2, max_frags=5)
    #processed_df.to_csv('../output/nano_dict_raw_processed.csv')
    pass
