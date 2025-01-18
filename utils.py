import os, urllib
import pandas as pd
from Bio import SeqIO
import numpy as np
import networkx as nx
import tarfile
import functools, signal


def read_fasta(filename):
    # read seq from fasta file
    all_ids = []
    all_seqs = []
    for seq_record in SeqIO.parse(filename, 'fasta'):
        all_ids.append(seq_record.id)
        all_seqs.append(seq_record.seq)
    return all_ids, all_seqs


def read_fasta_folder(path):
    # read seq from a folder of fasta files
    all_ids = []
    all_seqs = []
    file_list = [i for i in os.listdir(path) if (i.endswith('.fasta') or i.endswith('.fa'))]
    for file in file_list:
        ids, seqs = read_fasta(os.path.join(path, file))
        all_ids = all_ids + ids
        all_seqs = all_seqs + seqs
    return all_ids, all_seqs


def ReadFromTar(filename, temp_path='./_tmp'):
    no_path_label = False
    if not os.path.isdir(temp_path):
        no_path_label = True
    if type(filename) is list:
        file_list = filename
    else:
        file_list = [filename]
    for file in file_list:
        with tarfile.open(os.path.join(os.getcwd(), file), 'r:gz') as tar:
            tar.extractall(temp_path)
        extract_filename = file.split('.')[-3].split('/')[-1] + '_heads.pkl'
        extracted_file_path = os.path.join(temp_path, extract_filename)
        read_matrix = pd.read_pickle(extracted_file_path)
        os.remove(extracted_file_path)
        if no_path_label:
            os.rmdir(temp_path)
    return read_matrix


def PartNo(comm_dict):
    part_no = {}
    for i, partition in enumerate(comm_dict):
        for j in partition:
            part_no[j] = i
    return part_no


def SubgraphConnected(graph):
    subgraphs = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
    return subgraphs


def ListFromCluster(clusters, length=None):
    if length is None:
        length = np.max([np.max(list(i)) for i in clusters]) + 1
    list_clusters = np.zeros(length, dtype=int)
    for i, cl in enumerate(clusters):
        list_clusters[np.array(list(cl))] = i
    return list_clusters


def BatchListFromClusters(communities, length=None):
    cluster_lists = []
    communities_flatten = communities.reshape(-1)
    for clusters in communities_flatten:
        cluster_lists.append(ListFromCluster(clusters, length))
    return np.array(cluster_lists)


def RebuildGraph(cluster_list, length=None):
    if length is None:
        graph_size = len(cluster_list)
    else:
        graph_size = length
    num_blocks = max(cluster_list)
    graph = np.zeros((graph_size, graph_size))
    #for i in range(num_blocks):
    #    graph[np.where(cluster_list==i), np.where(cluster_list==i)] = 1
    clusters, clu_freq = np.unique(cluster_list, return_counts=True)[0], np.unique(cluster_list, return_counts=True)[1]
    clusters_sorted = clusters[np.argsort(clu_freq)]
    cluster_rank_dict = {i: np.where(clusters_sorted == i)[0] for i in clusters}
    #print(clusters_sorted)
    for i in range(graph_size):
        for j in range(graph_size):
            if cluster_list[i] == cluster_list[j]:
                graph[i, j] = cluster_list[i] + 1
                # graph[i, j] = 255 * (1-0.5*(cluster_list[i])/num_blocks)
                # graph[i, j] = cluster_rank_dict[cluster_list[i]]//10
    #graph[0, 0]  = 0
    return graph


def RebuildGraphFromClusters(communities, length=None):
    cluster_lists = BatchListFromClusters(communities, length=length)
    list_graphs = np.array([RebuildGraph(cluster_list, length=length).reshape(-1) for cluster_list in cluster_lists])
    #print([np.max(list(i)) for i in cluster[0,1]])
    return list_graphs


def FlattenGraphToGraph(flatten_graph):
    return flatten_graph.reshape(int(flatten_graph.shape[0] ** 0.5), -1)


def GraphToClusters(graph_rebuilt):
    list_clusters = []
    for i in np.delete(np.unique(graph_rebuilt), 0):
        # print(i, np.where(graph_rebuilt == i))
        list_clusters.append(set(np.unique(np.where(graph_rebuilt == i))))
    return list_clusters


def TimeOut(sec):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func

    return decorator


def ResidueClassification(sequence, mode=3):
    #
    # input: sequence of residues
    # output:
    class0 = dict.fromkeys(['A', 'V', 'L', 'I', 'M', 'P', 'F', 'W', 'G'], 'h')
    class0.update(dict.fromkeys(['S', 'T', 'Y', 'C', 'N', 'Q'], 'p'))
    class0.update(dict.fromkeys(['D', 'E'], 'a'))
    class0.update(dict.fromkeys(['H', 'K', 'R'], 'b'))

    class1 = dict.fromkeys(['A', 'V', 'L', 'I', 'M', 'P', 'G'], 'h')
    class1.update(dict.fromkeys(['F', 'W'], '@'))
    class1.update(dict.fromkeys(['S', 'T'], 'o'))
    class1.update(dict.fromkeys(['N', 'Q'], 'n'))
    class1.update(dict.fromkeys(['D', 'E'], 'a'))
    class1.update(dict.fromkeys(['K', 'R'], 'b'))
    class1['Y'] = 'Y';
    class1['C'] = 'C';
    class1['H'] = 'H'
    # IVL, M, AG / IVL, M, A, G
    class2 = dict.fromkeys(['A', 'V', 'L', 'I', 'M', 'G'], 'h')
    class2.update(dict.fromkeys(['F', 'W'], '@'))
    class2.update(dict.fromkeys(['S', 'T'], 'o'))
    class2.update(dict.fromkeys(['N', 'Q'], 'n'))
    class2.update(dict.fromkeys(['D', 'E'], 'a'))
    class2.update(dict.fromkeys(['K', 'R'], 'b'))
    class2['Y'] = 'Y';
    class2['C'] = 'C';
    class2['H'] = 'H';
    class2['P'] = 'P'

    class3 = dict.fromkeys(['I', 'V', 'L'], 'h')
    class3.update(dict.fromkeys(['A', 'G'], 's'))
    class3.update(dict.fromkeys(['F', 'W'], '@'))
    class3.update(dict.fromkeys(['S', 'T'], 'o'))
    class3.update(dict.fromkeys(['N', 'Q'], 'n'))
    class3.update(dict.fromkeys(['D', 'E'], 'a'))
    class3.update(dict.fromkeys(['K', 'R'], 'b'))
    class3['M'] = 'M'
    class3['Y'] = 'Y'
    class3['C'] = 'C'
    class3['H'] = 'H'
    class3['P'] = 'P'

    class4 = dict.fromkeys(['I', 'V', 'L'], 'h')
    class4.update(dict.fromkeys(['F', 'W'], '@'))
    class4.update(dict.fromkeys(['S', 'T'], 'o'))
    class4.update(dict.fromkeys(['N', 'Q'], 'n'))
    class4.update(dict.fromkeys(['D', 'E'], 'a'))
    class4.update(dict.fromkeys(['K', 'R'], 'b'))
    class4['A'] = 'A'
    class4['G'] = 'G'
    class4['M'] = 'M'
    class4['Y'] = 'Y'
    class4['C'] = 'C'
    class4['H'] = 'H'
    class4['P'] = 'P'

    restype_list = [i for i in 'IVLFWSTNQDEKRAGMYCHP']
    class5 = {i:i for i in restype_list}

    dictionary = [class0, class1, class2, class3, class4, class5]
    return ''.join([dictionary[mode][i] for i in sequence])


def DiscardHead(head_cls_dict, types_to_discard=['offset', 'keyres']):
    all_to_discard = []
    discard_mat = np.zeros((33, 20))
    for types in types_to_discard:
        all_to_discard += head_cls_dict[types]
    for i in all_to_discard:
        discard_mat[int(i.split('H')[0][1:]), int(i.split('H')[1])] = 1
    return discard_mat
