# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:59:12 2020

@author: fangjy
"""
import os
import pandas
import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from utils.load_data import load_data, load_AN
from utils.util import calculate_SE_kernel, tuple_to_sparse_matrix, sparse_normalize_adj

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import csv


class Graph(object):

    def __init__(self, dataset_str, path="data", exp_type="link", normalize_feature=False,
                 K_hop=1, with_feature=True, label_ratio=None):
        """
        :param dataset_str:
        :param path:
        :param exp_type: experiment type: link (link prediction) or classification
        """
        assert exp_type in ["link", "classification"] # exp_type must be link or classification

        if dataset_str in ['cora', 'citeseer', 'pubmed'] and label_ratio is None:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_data(dataset_str, path)
        else:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_AN(dataset_str, path, ratio=label_ratio)

        if exp_type == "link":
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = self.make_train_test_edges(adj)
            self.adj = sparse_normalize_adj(adj_train, K_hop)
            self.train_edges = train_edges
            self.val_edges = val_edges
            self.val_edges_false = val_edges_false
            self.test_edges = test_edges
            self.test_edges_false = test_edges_false

        else:
            # sparse matrix (tuple representation: indices, values, shape)
            self.adj = sparse_normalize_adj(adj, K_hop)

        if not with_feature:
            features = np.eye(features.shape[0], dtype=features.dtype)

        self.feature = self.get_tfidf_feature(features, dataset_str, normalize_feature)  # numpy array

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.graph = graph
        self.need_cal_SE_prior = True
        self.train_edge_dict = None
        self.n_nodes = self.feature.shape[0]

    def get_tfidf_feature(self, feature, dataset_str, normalize=False):

        if sp.issparse(feature):
            feature = feature.todense()

        if dataset_str != "pubmed":
            transformer = TfidfTransformer(smooth_idf=True)
            feature = transformer.fit_transform(feature).toarray()

        if normalize:
            norm = np.linalg.norm(feature, axis=1, keepdims=True)
            feature = feature / np.where(norm == 0., 1, norm)

        return feature

    def get_K_neighbors(self, K):

        if self.need_cal_SE_prior:
            # calculate prior covariance between nodes using Square Exponential Kernel
            self.SE_prior = calculate_SE_kernel(self.feature, variance=1.0, length_scale=1.0, n_fold=100)  # numpy array
            print("get prior covariance")
            self.need_cal_SE_prior = False

        neighbors = np.zeros([self.n_nodes, K], dtype=np.int32)

        for i in range(self.n_nodes):

            j = 0
            for neighbor in self.graph[i][:K]:
                neighbors[i, j] = neighbor
                j += 1

            if j < K:
                prior_neighbors = np.argsort(-1 * self.SE_prior[i])
                for neighbor in prior_neighbors:
                    if neighbor == i or neighbor in self.graph[i]:
                        continue
                    neighbors[i, j] = neighbor
                    j += 1
                    if j >= K:
                        break

        return neighbors

    def get_K_neighbors_prior(self, K):

        if self.need_cal_SE_prior:
            # calculate prior covariance between nodes using Square Exponential Kernel
            self.SE_prior = calculate_SE_kernel(self.feature, variance=1.0, length_scale=1.0, n_fold=100)  # numpy array
            print("get prior covariance")
            self.need_cal_SE_prior = False

        neighbors = np.zeros([self.n_nodes, K], dtype=np.int32)

        for i in range(self.n_nodes):

            j = 0
            prior_neighbors = np.argsort(-1 * self.SE_prior[i])
            for neighbor in prior_neighbors:
                if neighbor == i:
                    continue
                neighbors[i, j] = neighbor
                j += 1
                if j >= K:
                    break

        return neighbors

    def get_adj_label(self):

        indices, values, shape = self.adj
        values = np.array([1.] * len(values))
        label = tuple_to_sparse_matrix(indices, values, shape)
        label = label.toarray()

        return label

    def make_train_test_edges(self, adj, p_val=0.05, p_test=0.10):
        """
        adj is an adjacant matrix (scipy sparse matrix)

        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
        adj_train : training adjacant matrix
        train_edges : array indicating the training edges
        val_edges : array indicating the validation edges
        val_edge_false: array indicating the false edges in validation dataset
        """
        adj_row = adj.nonzero()[0]
        adj_col = adj.nonzero()[1]

        # get deges from adjacant matrix
        edges = []
        edges_dic = {}
        for i in range(len(adj_row)):
            edges.append([adj_row[i], adj_col[i]])
            edges_dic[(adj_row[i], adj_col[i])] = 1

        # split the dataset into training,validation and test dataset
        num_test = int(np.floor(len(edges) * p_test))
        num_val = int(np.floor(len(edges) * p_val))
        all_edge_idx = np.arange(len(edges))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        train_edge_idx = all_edge_idx[(num_val + num_test):]

        edges = np.asarray(edges)
        test_edges = edges[test_edge_idx]  # numpy array
        val_edges = edges[val_edge_idx]  # numpy array
        train_edges = edges[train_edge_idx]  # numpy array

        test_edges_false = []
        val_edges_false = []
        false_edges_dic = {}
        while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
            i = np.random.randint(0, adj.shape[0])
            j = np.random.randint(0, adj.shape[0])
            if (i, j) in edges_dic:
                continue
            if (j, i) in edges_dic:
                continue
            if (i, j) in false_edges_dic:
                continue
            if (j, i) in false_edges_dic:
                continue
            else:
                false_edges_dic[(i, j)] = 1
                false_edges_dic[(j, i)] = 1

            if np.random.random_sample() > 0.333:
                if len(test_edges_false) < num_test:
                    test_edges_false.append([i, j])
                else:
                    if len(val_edges_false) < num_val:
                        val_edges_false.append([i, j])
            else:
                if len(val_edges_false) < num_val:
                    val_edges_false.append([i, j])
                else:
                    if len(test_edges_false) < num_test:
                        test_edges_false.append([i, j])

        val_edges_false = np.asarray(val_edges_false)
        test_edges_false = np.asarray(test_edges_false)
        data = np.ones(train_edges.shape[0], dtype=adj.dtype)
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        # adj_train = adj_train + adj_train.T
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

    def get_all_false_edges(self):

        false_edges = []

        train_edges_dict = defaultdict(set)
        for edge in self.train_edges:
            train_edges_dict[edge[0]].add(edge[1])

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if j in train_edges_dict[i]:
                    continue
                false_edges.append((i, j))

        false_edges = np.asarray(false_edges)

        return false_edges

    def get_false_edges(self, num_false_edges):

        if self.train_edge_dict is None:

            train_edges_dict = defaultdict(set)

            for edge in self.train_edges:
                train_edges_dict[edge[0]].add(edge[1])
                train_edges_dict[edge[1]].add(edge[0])

            self.train_edge_dict = train_edges_dict

        false_edges = []

        while num_false_edges > 0:
            i = np.random.randint(0, self.n_nodes)
            j = np.random.randint(0, self.n_nodes)
            if j in self.train_edge_dict[i] or i in self.train_edge_dict[j]:
                continue
            false_edges.append((i, j))
            num_false_edges = num_false_edges - 1

        false_edges = np.asarray(false_edges)

        return false_edges


# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Dataset(object):
    def __init__(self, name, N, D, type, data_path='/data/'):
        self.data_path = data_path
        self.name, self.N, self.D = name, N, D
        assert type in ['regression', 'classification', 'multiclass']
        self.type = type

    def csv_file_path(self, name):
        return '{}{}.csv'.format(self.data_path, name)

    def read_data(self):
        data = pandas.read_csv(self.csv_file_path(self.name),
                               header=None, delimiter=',').values
        return {'X':data[:, :-1], 'Y':data[:, -1, None]}

    def download_data(self):
        NotImplementedError

    def get_data(self, seed=0, split=0, prop=0.9):
        path = self.csv_file_path(self.name)
        if not os.path.isfile(path):
            self.download_data()

        full_data = self.read_data()
        split_data = self.split(full_data, seed, split, prop)
        split_data = self.normalize(split_data, 'X')

        if self.type is 'regression':
            split_data = self.normalize(split_data, 'Y')

        return split_data

    def split(self, full_data, seed, split, prop):
        ind = np.arange(self.N)

        np.random.seed(seed + split)
        np.random.shuffle(ind)

        n = int(self.N * prop)

        X = full_data['X'][ind[:n], :]
        Xs = full_data['X'][ind[n:], :]

        Y = full_data['Y'][ind[:n], :]
        Ys = full_data['Y'][ind[n:], :]

        return {'X': X, 'Xs': Xs, 'Y': Y, 'Ys': Ys}

    def normalize(self, split_data, X_or_Y):
        m = np.average(split_data[X_or_Y], 0)[None, :]
        s = np.std(split_data[X_or_Y + 's'], 0)[None, :] + 1e-6

        if X_or_Y == 'X':
            split_data[X_or_Y] = (split_data[X_or_Y] - m) / s
            split_data[X_or_Y + 's'] = (split_data[X_or_Y + 's'] - m) / s
        else:
            split_data[X_or_Y] = split_data[X_or_Y] - m
            split_data[X_or_Y + 's'] = split_data[X_or_Y + 's'] - m

        split_data.update({X_or_Y + '_mean': m.flatten()})
        split_data.update({X_or_Y + '_std': s.flatten()})

        return split_data


datasets = []
uci_base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


class Boston(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'boston', 506, 12
        self.type = 'regression'

    def download_data(self):
        url = '{}{}'.format(uci_base, 'housing/housing.data')

        data = pandas.read_fwf(url, header=None).values
        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Concrete(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'concrete', 1030, 8
        self.type = 'regression'

    def download_data(self):
        url = '{}{}'.format(uci_base, 'concrete/compressive/Concrete_Data.xls')

        data = pandas.read_excel(url).values
        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Energy(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'energy', 768, 8
        self.type = 'regression'

    def download_data(self):
        url = '{}{}'.format(uci_base, '00242/ENB2012_data.xlsx')

        data = pandas.read_excel(url).values
        data = data[:, :-1]

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Kin8mn(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'kin8nm', 8192, 8
        self.type = 'regression'

    def download_data(self):

        url = 'http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm'

        data = pandas.read_csv(url, header=None).values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Naval(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'naval', 11934, 12
        self.type = 'regression'

    def download_data(self):

        url = '{}{}'.format(uci_base, '00316/UCI%20CBM%20Dataset.zip')

        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pandas.read_fwf('/tmp/UCI CBM Dataset/data.txt', header=None).values
        data = data[:, :-1]

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Power(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'power', 9568, 4
        self.type = 'regression'

    def download_data(self):
        url = '{}{}'.format(uci_base, '00294/CCPP.zip')
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pandas.read_excel('/tmp/CCPP//Folds5x2_pp.xlsx').values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Protein(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'protein', 45730, 9
        self.type = 'regression'

    def download_data(self):

        url = '{}{}'.format(uci_base, '00265/CASP.csv')

        data = pandas.read_csv(url).values

        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class WineRed(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'wine_red', 1599, 11
        self.type = 'regression'

    def download_data(self):

        url = '{}{}'.format(uci_base, 'wine-quality/winequality-red.csv')

        data = pandas.read_csv(url, delimiter=';').values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class WineWhite(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'wine_white', 4898, 12
        self.type = 'regression'

    def download_data(self):

        url = '{}{}'.format(uci_base, 'wine-quality/winequality-white.csv')

        data = pandas.read_csv(url, delimiter=';').values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class RegressionDatasets(object):

    def __init__(self, data_path='data/'):

        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        datasets = []

        datasets.append(Boston())
        datasets.append(Concrete())
        datasets.append(Energy())
        datasets.append(Kin8mn())
        datasets.append(Naval())
        datasets.append(Power())
        datasets.append(Protein())
        datasets.append(WineRed())
        datasets.append(WineWhite())

        self.all_datasets = {}
        for d in datasets:
            d.data_path = data_path
            self.all_datasets.update({d.name : d})


class SmallGraph(object):

    def __init__(self, dataset_str, path="data", K_hop=1):

        g = nx.read_gml("{}/{}/{}.gml".format(path, dataset_str, dataset_str), label=None)
        adj = nx.to_numpy_array(g)
        self.graph = self.get_graph(adj)

        all_edges = self.make_train_test_edges(adj)
        self.adj = sparse_normalize_adj(all_edges[0], K_hop)
        self.train_edges = all_edges[1]
        self.val_edges = all_edges[2]
        self.test_edges = all_edges[4]

        self.n_nodes = adj.shape[0]
        self.feature = np.eye(self.n_nodes, dtype=np.float64)

    def get_graph(self, adj):

        adj_row = adj.nonzero()[0]
        adj_col = adj.nonzero()[1]

        graph = defaultdict(list)

        for i in range(len(adj_row)):
            graph[adj_row[i]].append(adj_col[i])

        return graph

    def make_train_test_edges(self, adj, p_val=0.05, p_test=0.10):
        """
        adj is an adjacant matrix (scipy sparse matrix)

        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
        adj_train : training adjacant matrix
        train_edges : array indicating the training edges
        val_edges : array indicating the validation edges
        val_edge_false: array indicating the false edges in validation dataset
        """
        adj_row = adj.nonzero()[0]
        adj_col = adj.nonzero()[1]

        # get deges from adjacant matrix
        edges = []
        edges_dic = {}
        for i in range(len(adj_row)):
            edges.append([adj_row[i], adj_col[i]])
            edges_dic[(adj_row[i], adj_col[i])] = 1

        # split the dataset into training,validation and test dataset
        num_test = int(np.floor(len(edges) * p_test))
        num_val = int(np.floor(len(edges) * p_val))
        all_edge_idx = np.arange(len(edges))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        train_edge_idx = all_edge_idx[(num_val + num_test):]

        edges = np.asarray(edges)
        test_edges = edges[test_edge_idx]  # numpy array
        val_edges = edges[val_edge_idx]  # numpy array
        train_edges = edges[train_edge_idx]  # numpy array

        test_edges_false = []
        val_edges_false = []
        false_edges_dic = {}
        while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
            i = np.random.randint(0, adj.shape[0])
            j = np.random.randint(0, adj.shape[0])
            if (i, j) in edges_dic:
                continue
            if (j, i) in edges_dic:
                continue
            if (i, j) in false_edges_dic:
                continue
            if (j, i) in false_edges_dic:
                continue
            else:
                false_edges_dic[(i, j)] = 1
                false_edges_dic[(j, i)] = 1

            if np.random.random_sample() > 0.333:
                if len(test_edges_false) < num_test:
                    test_edges_false.append([i, j])
                else:
                    if len(val_edges_false) < num_val:
                        val_edges_false.append([i, j])
            else:
                if len(val_edges_false) < num_val:
                    val_edges_false.append([i, j])
                else:
                    if len(test_edges_false) < num_test:
                        test_edges_false.append([i, j])

        val_edges_false = np.asarray(val_edges_false)
        test_edges_false = np.asarray(test_edges_false)
        data = np.ones(train_edges.shape[0], dtype=adj.dtype)
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        # adj_train = adj_train + adj_train.T
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

    def get_K_neighbors(self, K):

        neighbors = []

        for i in range(self.n_nodes):

            node_neighbors = []
            candidate_neighbors = np.random.permutation(self.graph[i]).tolist()

            j = 0
            for neighbor in candidate_neighbors:

                if neighbor in node_neighbors:
                    continue

                node_neighbors.append(neighbor)
                j += 1
                if j >= K:
                    break

                candidate_neighbors.extend(np.random.permutation(self.graph[neighbor]).tolist())

            neighbors.append(node_neighbors)

        return np.asarray(neighbors, dtype=np.int32)

    def get_all_false_edges(self):

        false_edges = []

        train_edges_dict = defaultdict(set)
        for edge in self.train_edges:
            train_edges_dict[edge[0]].add(edge[1])

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if j in train_edges_dict[i]:
                    continue
                false_edges.append((i, j))

        false_edges = np.asarray(false_edges)

        return false_edges






