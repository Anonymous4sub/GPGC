import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, path):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/{}/ind.{}.{}".format(path, dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/{}/ind.{}.test.index".format(path, dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def read_label(labels, n_train):

    label_list = list()
    idx = 0
    for label in labels:
        l = int(label.strip())
        label_list.append(l)
        idx += 1

    num_nodes = idx
    num_label = len(np.unique(label_list))

    if np.min(label_list) > 0:
        label_list = np.asarray(label_list) - 1  # -1 to let label start from 0
    else:
        label_list = np.asarray(label_list)

    label_dict = defaultdict(list)  # used for train, val, test split
    for i, l in enumerate(label_list):
        label_dict[l].append(i)

    all_label = np.zeros((num_nodes, num_label), dtype=np.float64)
    for i in range(num_nodes):
        all_label[i, label_list[i]] = 1.

    # number of each class in test set
    if isinstance(n_train, float):
        num_train = num_nodes * n_train
    num_val = 500
    num_test = 1000
    num_train_dict = {}
    num_val_dict = {}
    num_test_dict = {}

    for k, v in label_dict.items():
        label_ratio = len(v) / num_nodes
        if isinstance(n_train, float):
            num_train_dict[k] = int(np.round(num_train * label_ratio, 0))
        else:
            num_train_dict[k] = n_train

        num_val_dict[k] = int(np.round(num_val * label_ratio, 0))
        num_test_dict[k] = int(np.round(num_test * label_ratio, 0))

    idx_train = []
    idx_val = []
    idx_test = []
    for l in range(num_label):
        node_index = np.random.permutation(label_dict[l])
        idx_train.extend(node_index[:num_train_dict[l]])
        idx_val.extend(node_index[num_train_dict[l]:num_train_dict[l]+num_val_dict[l]])
        idx_test.extend(node_index[-num_test_dict[l]:])

    idx_train = np.sort(idx_train)
    idx_val = np.sort(idx_val)
    idx_test = np.sort(idx_test)

    train_mask = sample_mask(idx_train, all_label.shape[0])
    val_mask = sample_mask(idx_val, all_label.shape[0])
    test_mask = sample_mask(idx_test, all_label.shape[0])

    y_train = np.zeros(all_label.shape)
    y_val = np.zeros(all_label.shape)
    y_test = np.zeros(all_label.shape)
    y_train[train_mask, :] = all_label[train_mask, :]
    y_val[val_mask, :] = all_label[val_mask, :]
    y_test[test_mask, :] = all_label[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def read_label_v2(labels, n_train):

    label_list = list()
    idx = 0
    for label in labels:
        l = int(label.strip())
        label_list.append(l)
        idx += 1

    num_nodes = idx
    num_label = len(np.unique(label_list))

    if np.min(label_list) > 0:
        label_list = np.asarray(label_list) - 1  # -1 to let label start from 0
    else:
        label_list = np.asarray(label_list)

    all_label = np.zeros((num_nodes, num_label), dtype=np.float64)
    for i in range(num_nodes):
        all_label[i, label_list[i]] = 1.

    if isinstance(n_train, float):
        num_train = int(np.round(num_nodes * n_train))
    else:
        num_train = num_label * n_train

    idx_rand = np.random.permutation(np.arange(num_nodes))
    idx_train = np.sort(idx_rand[:num_train])
    idx_val = np.sort(idx_rand[num_train:num_train+500])
    idx_test = np.sort(idx_rand[-1000:])

    train_dict = {}
    for idx in idx_train:
        l = label_list[idx]
        train_dict[l] = train_dict.get(l, 0) + 1
    print(train_dict)

    train_mask = sample_mask(idx_train, all_label.shape[0])
    val_mask = sample_mask(idx_val, all_label.shape[0])
    test_mask = sample_mask(idx_test, all_label.shape[0])

    y_train = np.zeros(all_label.shape)
    y_val = np.zeros(all_label.shape)
    y_test = np.zeros(all_label.shape)
    y_train[train_mask, :] = all_label[train_mask, :]
    y_val[val_mask, :] = all_label[val_mask, :]
    y_test[test_mask, :] = all_label[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_AN(dataset, path, ratio=20):

    # ratio: integer or float

    edge_file = open("{}/{}/{}.edge".format(path, dataset, dataset), 'r')
    attri_file = open("{}/{}/{}.node".format(path, dataset, dataset), 'r')
    label_file = open("{}/{}/{}.label".format(path, dataset, dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()

    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())

    print("dataset:{}, node_num:{},edge_num:{}, attribute_"
          "nunber:{}".format(dataset, node_num, edge_num, attribute_number))

    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []
    graph = defaultdict(list)

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
        graph[node1].append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))

    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))

    print("load_data finished!")

    # load labels
    y_train, y_val, y_test, train_mask, val_mask, test_mask = read_label(label_file.readlines(), n_train=ratio)

    return adj, attribute, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph
