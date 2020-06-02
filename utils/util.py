# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:04:41 2020

@author: fangjy
"""
import scipy.sparse as sp
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from networkx.algorithms.community import greedy_modularity_communities


def tuple_to_sparse_matrix(indices, values, shape):
    row = [idx[0] for idx in indices]
    col = [idx[1] for idx in indices]

    return sp.csc_matrix((values, (row, col)), shape=shape)


def pca(X, out_dim):

    dim = X.shape[1]
    if dim > 2:
        p = PCA(n_components=out_dim)
        return p.fit_transform(X)
    else:
        return X


def get_sigma_ii_idx(N):
    idx = []
    # for out_idx in range(num_outputs):
    for n_idx in range(N):
        # idx.append([out_idx, n_idx, n_idx])
        idx.append([n_idx, n_idx])

    return idx


def get_sigma_ia_idx(N, neighbors):
    idx = []
    # for out_idx in range(num_outputs):
    for n_idx in range(N):
        idx.extend([[n_idx, neighbor_idx] for neighbor_idx in neighbors[n_idx]])

    return idx


def get_sigma_aa_idx(N, neighbors):
    idx = []
    # for out_idx in range(num_outputs):
    for n_idx in range(N):
        for a in neighbors[n_idx]:
            for b in neighbors[n_idx]:
                idx.append([a, b])

    return idx


def get_mean_ia_idx(N, neighbors):
    idx = []
    for n_idx in range(N):
        for ne in neighbors[n_idx]:
            idx.append([ne])

    return idx


def tile_indices(num_outputs, indices):

    indices = np.array(indices)
    length = indices.shape[0]
    indices_tile = np.tile(indices, [num_outputs, 1])

    for i in range(num_outputs):
        indices_tile[i*length:(i+1)*length, 0] = i

    return indices_tile.tolist()


def calculate_SE_kernel(X, variance=1.0, length_scale=1.0, n_fold=100):
    # calculate prior covariance of X
    Y = X.copy()

    X_folds = []
    num_folds = X.shape[0] // 100
    for i in range(num_folds):
        if i == num_folds - 1:
            X_folds.append(X[i*n_fold:])
        else:
            X_folds.append(X[i*n_fold:(i+1)*n_fold])
    print("get X_folds")

    kernel_matrix_folds = []
    # X = np.expand_dims(X, axis=-2)
    Y = np.expand_dims(Y, axis=-3)

    for x in X_folds:
        x = np.expand_dims(x, axis=-2)
        # calculate distance
        distance = np.sum(np.square(x - Y), axis=-1)
        # calculate kernel matrix
        sub_kernel_matrix = variance * np.exp(-0.5 * distance / length_scale)
        kernel_matrix_folds.append(sub_kernel_matrix)

    kernel_matrix = np.concatenate(kernel_matrix_folds, axis=0)
    print("finish calculating kernel matrix")

    return kernel_matrix


def sigmoid(x):
    return 1. / (1+np.exp(-1*x))


def roc_ap_score(embedding, edge_pos, edge_neg, logits=True):

    head_pos_embed = embedding[edge_pos[:, 0]]
    tail_pos_embed = embedding[edge_pos[:, 1]]

    head_neg_embed = embedding[edge_neg[:, 0]]
    tail_neg_embed = embedding[edge_neg[:, 1]]

    preds_pos = sigmoid(np.sum(np.multiply(head_pos_embed, tail_pos_embed), axis=1))
    preds_neg = sigmoid(np.sum(np.multiply(head_neg_embed, tail_neg_embed), axis=1))

    preds_all = np.hstack([preds_pos, preds_neg])
    true_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    auc_score = roc_auc_score(true_all, preds_all)
    ap_score = average_precision_score(true_all, preds_all)

    return auc_score, ap_score


def get_roc_ap_score(F, Fmean, edges_pos, edges_neg):
    """
    :param F, Fmean: [S, N, D]
    """
    num_samples = F.shape[0]
    auc_sample = []
    ap_sample = []
    auc_mean = []
    ap_mean = []

    for i in range(num_samples):

        embedding = F[i]
        # reconstruction = np.dot(embedding, np.transpose(embedding))
        auc, ap = roc_ap_score(embedding, edges_pos, edges_neg)
        auc_sample.append(auc)
        ap_sample.append(ap)

        embedding = Fmean[i]
        # reconstruction = np.dot(embedding, np.transpose(embedding))
        auc, ap = roc_ap_score(embedding, edges_pos, edges_neg)
        auc_mean.append(auc)
        ap_mean.append(ap)

    return np.mean(auc_sample), np.mean(ap_sample), np.mean(auc_mean), np.mean(ap_mean)


def get_classification_label(graph, GCGP_X=False):

    num_classes = graph.y_train.shape[1]
    y_train = np.asarray(graph.y_train[graph.train_mask], dtype=np.int32)
    y_train = np.reshape(np.sum(np.tile(np.arange(num_classes), (np.sum(graph.train_mask), 1)) * y_train, axis=1), (-1,1))
    y_val = np.asarray(graph.y_val[graph.val_mask], dtype=np.int32)
    y_val = np.reshape(np.sum(np.tile(np.arange(num_classes), (np.sum(graph.val_mask), 1)) * y_val, axis=1), (-1, 1))
    y_test = np.asarray(graph.y_test[graph.test_mask], dtype=np.int32)
    y_test = np.reshape(np.sum(np.tile(np.arange(num_classes), (np.sum(graph.test_mask), 1)) * y_test, axis=1), (-1, 1))

    idx_train = np.arange(graph.y_train.shape[0])[graph.train_mask]

    if GCGP_X:
        y_train = np.concatenate((y_train, y_val), axis=0)
        idx_val = np.arange(graph.y_train.shape[0])[graph.val_mask]
        idx_train = np.concatenate((idx_train, idx_val), axis=-1)

    return y_train, y_val, y_test, idx_train


def classification_acc(F, Fmean, mask, label):

    n_samples = F.shape[0]
    accs_sample = []
    accs_mean = []

    for i in range(n_samples):

        acc_sample = np.sum(np.argmax(F[i][mask], axis=1) == label.flatten()) * 100. / len(label)
        acc_mean = np.sum(np.argmax(Fmean[i][mask], axis=1) == label.flatten()) * 100. / len(label)
        accs_sample.append(acc_sample)
        accs_mean.append(acc_mean)

    return np.mean(accs_sample), np.mean(accs_mean)


def sparse_to_tuple(sparse_mx):

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


def sparse_normalize_adj(adj, K_hop=1):

    adj = sp.coo_matrix(adj + np.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    normalized_adj_hops = normalized_adj.copy()

    for _ in range(K_hop-1):
        normalized_adj_hops = normalized_adj_hops.dot(normalized_adj)

    tuple_normalized_adj = sparse_to_tuple(normalized_adj_hops)

    return tuple_normalized_adj


def reg_data_processing(data, K_hop):

    # data: dict, ['X', 'Xs', 'Y', 'Ys', 'X_mean', 'X_std', 'Y_mean', 'Y_std']

    feature = np.concatenate((data['X'], data['Xs']), axis=0)
    num_train = data['X'].shape[0]
    idx_train = np.arange(feature.shape[0])[:num_train]

    k = 1
    while True:
        adj = kneighbors_graph(feature, k, mode='connectivity', include_self=False)
        G = nx.Graph(adj)
        if nx.is_connected(G):
            break
        k += 1

    adj = sparse_normalize_adj(adj, K_hop=K_hop)

    return feature, adj, idx_train, k


def get_K_neighbors_prior(K, prior):

    n_nodes = prior.shape[0]

    neighbors = np.zeros([n_nodes, K], dtype=np.int32)

    for i in range(n_nodes):

        j = 0
        prior_neighbors = np.argsort(-1 * prior[i])
        for neighbor in prior_neighbors:
            if neighbor == i:
                continue
            neighbors[i, j] = neighbor
            j += 1
            if j >= K:
                break

    return neighbors


def regression_test_nll(mean_SND, var_SND, Ys, Y_std):

    S = mean_SND.shape[0]
    mean_ND = np.average(mean_SND, 0)

    MSE = np.mean(np.square(mean_ND - Ys))
    logps = norm.logpdf(np.repeat(Ys[None, :, :], S, axis=0), mean_SND, np.sqrt(var_SND))
    test_nll = np.mean(logsumexp(logps, axis=0) - np.log(S))

    return MSE, test_nll


"""
    mean_SND = mean_SND * Y_std
    std_SND = var_SND ** 0.5 * var_SND
    Ys = Ys * Y_std

    test_err = np.average(Y_std * np.mean((Ys - mean_ND) ** 2.0) ** 0.5)
    logps = norm.logpdf(np.repeat(Ys[None, :, :], S, axis=0), mean_SND, std_SND)
    test_nll_ND = logsumexp(logps, axis=0) - np.log(S)

    # test_nll_ND = logsumexp(norm.logpdf(Ys * Y_std, mean_SND * Y_std, var_SND ** 0.5 * Y_std), 0, b=1 / float(S))
    test_nll = np.average(test_nll_ND)
    
    return test_err, test_nll

ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
logps = norm.logpdf(np.repeat(Ys[None, :, :], self.ARGS.num_posterior_samples, axis=0), ms, np.sqrt(vs))
return logsumexp(logps, axis=0) - np.log(self.ARGS.num_posterior_samples)
"""


def link_likelihood(embedding, edges):

    head_embed = embedding[edges[:, 0]]
    tail_embed = embedding[edges[:, 1]]

    prob = sigmoid(np.sum(np.multiply(head_embed, tail_embed), axis=1)) + 1e-5

    return prob


def link_negative_log_likelihood(F, Fmean, edges):

    num_samples = F.shape[0]
    nll_sample = []
    nll_mean = []

    for i in range(num_samples):

        embedding = F[i]
        nll = - np.mean(np.log(link_likelihood(embedding, edges)))
        nll_sample.append(nll)

        embedding = Fmean[i]
        nll = - np.mean(np.log(link_likelihood(embedding, edges)))
        nll_mean.append(nll)

    return np.mean(nll_sample), np.mean(nll_mean)


def mean_std(string):

    result = np.array([float(s.strip()) for s in string.split(",")])
    n = len(result)

    mean = np.mean(result)

    std = np.sqrt(np.sum(np.square(result - mean)) / (n-1)) / np.sqrt(n)

    return mean, std


def tsne(array, n):

    dimen = array.shape[1]

    if dimen > 2:
        model = TSNE(n_components=n, random_state=0)
        np.set_printoptions(suppress=True)
        print("use TSNE...")
        return model.fit_transform(array)
    else:
        return array


def plot_scatter(ax, data, color_list):

    area = 5
    ax.set_aspect("equal")
    ax.scatter(data[:, 0], data[:, 1], s=area, c=color_list, alpha=1, marker=(9, 1, 30))
    # ax.scatter(data[:, 0], data[:, 1], s=area, alpha=1, marker=(9, 1, 30))

    """
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    """

    ax.set_xticks([])
    ax.set_yticks([])


def plot_feature(path, sample=False):

    dataset = path.split("/")[1]

    orig_feature = np.load("{}/orig_feature.npy".format(path))
    transformed_feature = np.load("{}/transformed_feature.npy".format(path))
    # transformed_feature = transformed_feature[0]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    orig_fea_tsne = tsne(orig_feature, 2)
    trans_fea_tsne = tsne(transformed_feature, 2)
    orig_fea_tsne = scaler.fit_transform(orig_fea_tsne)
    trans_fea_tsne = scaler.fit_transform(trans_fea_tsne)

    # get community
    g = nx.read_gml("{}/{}.gml".format(path, dataset), label=None)
    partition = greedy_modularity_communities(g)
    n_nodes = g.number_of_nodes()
    n_com = len(partition)
    start_id = np.min(g.nodes)
    com_dict = dict()
    for i in range(n_com):
        for node in partition[i]:
            com_dict[node - start_id] = i  # -1 because node id start from 1

    # get node color
    colors = ['red', 'blue', 'green', 'aqua', 'yellow', 'skyblue', 'purple', 'olive']
    color_list = []
    for node in range(n_nodes):
        color_list.append(colors[com_dict[node]])

    """
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(131)
    plot_scatter(ax, orig_fea_tsne, color_list)
    ax = fig.add_subplot(132)
    plot_scatter(ax, trans_fea_tsne, color_list)

    if sample is True:

        F = np.load("{}/F.npy".format(path))
        F_tsne = tsne(F, 2)
        F_tsne = scaler.fit_transform(F_tsne)

        ax = fig.add_subplot(133)
        plot_scatter(ax, F_tsne, color_list)
    
    """

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    plot_scatter(ax, trans_fea_tsne, color_list)

    fig.tight_layout()

    plt.savefig("{}/{}_feature.pdf".format(path, dataset), format='pdf', dpi=1000)
    plt.show()


def convert_to_csv(dataset, path):

    edge_file = open("{}/{}.edge".format(path, dataset), 'r')
    attri_file = open("{}/{}.node".format(path, dataset), 'r')
    label_file = open("{}/{}.label".format(path, dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    labels = label_file.readlines()

    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())

    print("dataset:{}, node_num:{},edge_num:{}, attribute_"
          "nunber:{}".format(dataset, node_num, edge_num, attribute_number))

    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row_orig = []
    adj_col_orig = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row_orig.append(node1)
        adj_col_orig.append(node2)

    # sort edges
    idx_sort = np.argsort(adj_row_orig)
    adj_row = []
    adj_col = []
    for idx in idx_sort:
        if adj_row_orig[idx] == adj_col_orig[idx]:
            continue
        adj_row.append(adj_row_orig[idx])
        adj_col.append(adj_col_orig[idx])

    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)

    label_list = list()
    for label in labels:
        l = int(label.strip())
        label_list.append(l)

    edge_array = np.zeros((len(adj_row), 2), dtype=np.int64)
    feature_array = np.zeros((node_num, attribute_number), dtype=np.int64)

    for i in range(len(adj_row)):
        edge_array[i, 0] = adj_row[i]
        edge_array[i, 1] = adj_col[i]
    print("get edge_array")

    for i in range(len(att_row)):
        feature_array[att_row[i], att_col[i]] = 1

    print("get feature array")

    print(edge_array.shape)
    print(feature_array.shape)

    # label
    unique_label = np.unique(label_list)
    label_dict = {}
    for l in unique_label:
        label_dict[l] = 'c{}'.format(l)

    label_str = []
    for l in label_list:
        label_str.append(label_dict[l])

    edge_list = pd.DataFrame(edge_array)
    node_feature = pd.DataFrame(feature_array)
    node_feature["subject"] = label_str

    edge_list.to_csv("{}_edgelist.csv".format(dataset), sep="\t", header=False, index=True)
    node_feature.to_csv("{}_nodefeature.csv".format(dataset), sep="\t", header=False, index=True)

    print("successfully saved!")

