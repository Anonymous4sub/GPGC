# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:56:00 2020

@author: fangjy
"""

import numpy as np
from scipy.cluster.vq import kmeans2
import tensorflow as tf

import gpflow
from gpflow import settings, autoflow, params_as_tensors, ParamList
from gpflow.mean_functions import Identity, Linear, Zero
from gpflow.features import InducingPoints
from gcgp.gckernel import SparseGraphConvolution, GraphConvolutionInducingpoints
from gpflow.models import Model

from utils.util import *
from gcgp.layers import svgp_layer


def get_kernels(base_kernels, input_dims):

    kernels = list()
    i = 0
    for kernel, in_dim in zip(base_kernels, input_dims):
        kernels.append(kernel(in_dim))
        tf.logging.info("========Settings of kernel {}========".format(i))
        tf.logging.info("input dimension: {}".format(in_dim))
        tf.logging.info("=====================================")
        i += 1

    return kernels


def get_graphconvolutionkernels(base_kernels, input_dims, base_kernel_dims, adj_sparse_tensor, gc_weight):
    # get graph convolution kernels
    kernels = list()
    i = 0
    for in_dim, kernel_dim in zip(input_dims, base_kernel_dims):
        # print(in_dim, out_dim)
        kernels.append(SparseGraphConvolution(in_dim, base_kernels[i], adj_sparse_tensor,
                                              gc_weight[i], kernel_dim, i))
        # print(self.kernels[i].base_kernel.input_dim)
        tf.logging.info("========Settings of kernel {}========".format(i))
        tf.logging.info("input dimension: {}".format(in_dim))
        tf.logging.info("base kernel input dimension: {}".format(kernel_dim))
        tf.logging.info("gc_weight: {}".format(gc_weight[i]))
        tf.logging.info("=====================================")

        i += 1

    return kernels


def init_layers(graph_adj, node_feature, kernels, n_layers, all_layers_dim, num_inducing,
                gc_kernel=True, mean_function="linear", white=False, q_diag=False):

    assert mean_function in ["linear", "zero"]  # mean function must be linear or zero

    layers = []

    # get initial Z
    sparse_adj = tuple_to_sparse_matrix(graph_adj[0], graph_adj[1], graph_adj[2])
    X_running = node_feature.copy()

    for i in range(n_layers):

        tf.logging.info("initialize {}th layer".format(i + 1))

        dim_in = all_layers_dim[i]
        dim_out = all_layers_dim[i + 1]

        conv_X = sparse_adj.dot(X_running)
        Z_running = kmeans2(conv_X, num_inducing[i], minit="points")[0]

        kernel = kernels[i]

        if gc_kernel and kernel.gc_weight:
            # Z_running = pca(Z_running, kernel.base_kernel.input_dim)  # 将维度降到和输出维度一致
            X_dim = X_running.shape[1]
            kernel_input_dim = kernel.base_kernel.input_dim
            if X_dim > kernel_input_dim:
                Z_running = pca(Z_running, kernel.base_kernel.input_dim)  # 将维度降到和输出维度一致
            elif X_dim < kernel_input_dim:
                Z_running = np.concatenate([Z_running, np.zeros((Z_running.shape[0], kernel_input_dim - X_dim))], axis=1)

        # print(type(Z_running))
        # print(Z_running)

        if dim_in > dim_out:
            _, _, V = np.linalg.svd(X_running, full_matrices=False)
            W = V[:dim_out, :].T
        elif dim_in < dim_out:
            W = np.concatenate([np.eye(dim_in), np.zeros((dim_in, dim_out - dim_in))], 1)

        if mean_function == "zero":
            mf = Zero()
        else:

            if dim_in == dim_out:
                mf = Identity()
            else:
                mf = Linear(W)
                mf.set_trainable(False)

        # self.Ku = Kuu(GraphConvolutionInducingpoints(Z_running), kernel, jitter=settings.jitter)
        # print("successfully calculate Ku")
        if gc_kernel:
            feature = GraphConvolutionInducingpoints(Z_running)
        else:
            feature = InducingPoints(Z_running)

        layers.append(svgp_layer(kernel, Z_running, feature, dim_out, mf, gc_kernel, white=white, q_diag=q_diag))

        if dim_in != dim_out:
            # Z_running = Z_running.dot(W)
            X_running = X_running.dot(W)

    return layers


class GCGP_base(Model):

    def __init__(self, X, adj, layers, sample, n_samples, K,
                 neighbors=None, loss_type="link_full", label=None,
                 pos_edges=None, neg_edges=None, idx_train=None,
                 linear_layer=False, n_split=None, name="GCGP_base", **kwargs):
        """
        :param X: tensor placeholder
        :param adj: sparse tensor placeholder
        :param label:
        :param layers:
        :param sample:
        :param n_samples:
        :param K:
        :param neighbors: n-array, [n_nodes, K]
        """
        Model.__init__(self, name=name, **kwargs)

        self.X = X
        self.adj = adj
        self.loss_type = loss_type
        self.label = label
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.idx_train = idx_train
        self.n_split = n_split
        self.linear_layer = linear_layer
        self.layers = ParamList(layers)

        self.sample = sample
        self.n_samples = n_samples
        self.K = K

        # indices for neighbor sampling
        self.neighbor_indices = [None] * 4

        if self.sample == "neighbor":
            assert neighbors is not None  # neighbors should not be None when sample="neighbor"
            self.update_neighbor_indices(neighbors.shape[0], neighbors)

        if self.loss_type == "classification":
            self.likelihood = gpflow.likelihoods.MultiClass(len(np.unique(self.label)))

        if self.loss_type == "regression":
            self.likelihood = gpflow.likelihoods.Gaussian(variance=0.1)

        if self.linear_layer:
            with tf.variable_scope("linear_weight"):
                self.W = tf.get_variable(name="linear_w", shape=[self.layers[-1].num_outputs, self.label.shape[1]],
                                         dtype=settings.float_type, initializer=tf.glorot_uniform_initializer())

    def calculate_loss(self, kl_scale=1.):

        if self.loss_type == "link_full":
            return -self._likelihood_full_link(), -self.E_log_likelihood, self.KL
        elif self.loss_type == "link":
            return -self._likelihood_link(), -self.E_log_likelihood, self.KL
        elif self.loss_type == "classification":
            return -self._likelihood_classification(kl_scale), -self.E_log_likelihood, self.KL
        elif self.loss_type == "classification_softmax":
            return -self._likelihood_classification_softmax(kl_scale), -self.E_log_likelihood, self.KL
        else:
            return -self._likelihood_regression(), -self.E_log_likelihood, self.KL

    def _likelihood_full_link(self):

        Fs, Fmeans, Fvars = self.get_forward_samples(self.n_samples)
        f_final_layer = Fs[-1]  # [n_samples, n_nodes, embedding_dim] or [S, N, D]

        # Expected log likelihood
        logits = tf.matmul(f_final_layer, tf.transpose(f_final_layer, (0, 2, 1)))  # [S, N, N]
        f_llh = lambda x: - tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=x))
        self.E_log_likelihood = tf.reduce_mean(tf.map_fn(f_llh, logits))

        # KL divergence
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        self.KL = KL / tf.cast(tf.shape(self.X)[0] ** 2, settings.float_type)

        elbo = self.E_log_likelihood - self.KL

        return elbo

    def _likelihood_link(self):

        def likelihood(embedding):

            head_pos_embed = tf.nn.embedding_lookup(embedding, self.pos_edges[:, 0])
            tail_pos_embed = tf.nn.embedding_lookup(embedding, self.pos_edges[:, 1])
            head_neg_embed = tf.nn.embedding_lookup(embedding, self.neg_edges[:, 0])
            tail_neg_embed = tf.nn.embedding_lookup(embedding, self.neg_edges[:, 1])

            pos_score = tf.reduce_sum(tf.multiply(head_pos_embed, tail_pos_embed), axis=1)
            neg_score = tf.reduce_sum(tf.multiply(head_neg_embed, tail_neg_embed), axis=1)
            pos_label = tf.ones_like(pos_score)
            neg_label = tf.zeros_like(neg_score)

            logits = tf.concat([pos_score, neg_score], axis=-1)
            label = tf.concat([pos_label, neg_label], axis=-1)

            llh = - tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))

            return llh

        Fs, Fmeans, Fvars = self.get_forward_samples(self.n_samples)
        f_final_layer = Fs[-1]  # [n_samples, n_nodes, embedding_dim] or [S, N, D]

        self.E_log_likelihood = tf.reduce_mean(tf.map_fn(likelihood, f_final_layer))

        # KL divergence
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        self.KL = KL / tf.cast(tf.shape(self.pos_edges)[0] + tf.shape(self.neg_edges)[0], settings.float_type)

        elbo = self.E_log_likelihood - self.KL

        return elbo

    def _likelihood_classification(self, kl_scale=1.):

        Fs, Fmeans, Fvars = self.get_forward_samples(self.n_samples)
        Fmean, Fvar = Fmeans[-1], Fvars[-1]

        f = lambda x: tf.gather(x, self.idx_train)
        Fmean_tr = tf.map_fn(f, Fmean)
        Fvar_tr = tf.map_fn(f, Fvar)

        E_log_likelihood = tf.reduce_mean(self.likelihood.variational_expectations(Fmean_tr[0], Fvar_tr[0], self.label))

        for i in range(self.n_samples):
            if i == 0: continue
            E_log_likelihood += tf.reduce_mean(self.likelihood.variational_expectations(Fmean_tr[i], Fvar_tr[i], self.label))

        self.E_log_likelihood = E_log_likelihood / tf.cast(tf.shape(Fmean)[0], settings.float_type)

        # KL divergence
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        self.KL = KL / tf.cast(tf.shape(self.X)[0], settings.float_type) * kl_scale

        elbo = self.E_log_likelihood - self.KL

        return elbo

    def _likelihood_classification_softmax(self, kl_scale=1.):

        def likelihood(logits):

            logits_train = tf.gather(logits, self.idx_train)
            llh = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=logits_train))
            return llh

        Fs, Fmeans, Fvars = self.get_forward_samples(self.n_samples)
        f_final_layer = Fs[-1]  # [n_samples, n_nodes, embedding_dim] or [S, N, D]

        if self.linear_layer:
            f_final_layer = tf.matmul(f_final_layer, self.W)

        self.E_log_likelihood = tf.reduce_mean(tf.map_fn(likelihood, f_final_layer))

        # KL divergence
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        # self.KL = KL / tf.cast(tf.shape(self.X)[0], settings.float_type) * kl_scale
        self.KL = KL / tf.cast(len(self.idx_train), settings.float_type) * kl_scale

        elbo = self.E_log_likelihood - self.KL

        return elbo

    def _likelihood_regression(self):

        Fs, Fmeans, Fvars = self.get_forward_samples(self.n_samples)
        Fmean, Fvar = Fmeans[-1], Fvars[-1]

        f = lambda x: tf.gather(x, self.idx_train)
        Fmean_tr = tf.map_fn(f, Fmean)
        Fvar_tr = tf.map_fn(f, Fvar)

        E_log_likelihood = tf.reduce_mean(self.likelihood.variational_expectations(Fmean_tr[0], Fvar_tr[0], self.label))

        for i in range(self.n_samples):
            if i == 0: continue
            E_log_likelihood += tf.reduce_mean(
                self.likelihood.variational_expectations(Fmean_tr[i], Fvar_tr[i], self.label))

        self.E_log_likelihood = E_log_likelihood / tf.cast(tf.shape(Fmean)[0], settings.float_type)

        # KL divergence
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        self.KL = KL / tf.cast(tf.shape(self.X)[0], settings.float_type)

        elbo = self.E_log_likelihood - self.KL

        return elbo

    def get_forward_samples(self, n_samples):

        sX = tf.tile(tf.expand_dims(self.X, 0), [n_samples, 1, 1])
        Fs, Fmeans, Fvars = [], [], []

        F = sX

        for (i, layer) in enumerate(self.layers):
            if self.sample == "diagonal":
                F, Fmean, Fvar = layer.sample_from_conditional(F, full_cov=False)
            elif self.sample == "full":
                F, Fmean, Fvar = layer.sample_from_conditional(F, full_cov=True)
            elif self.sample == "neighbor":
                F, Fmean, Fvar = layer.sample_from_neighbor_conditional(F, self.K, self.neighbor_indices, self.n_split)
            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    def update_neighbor_indices(self, n, neighbors):

        # sigma_ii_idx, sigma_ia_idx, sigma_aa_idx, mean_ia_idx
        # n = self.graph.n_nodes
        # neighbors = self.graph.get_K_neighbors(self.K)  # [ [k_neighbors], [] ]

        base_sigma_ii = get_sigma_ii_idx(n)
        base_sigma_ia = get_sigma_ia_idx(n, neighbors)
        base_sigma_aa = get_sigma_aa_idx(n, neighbors)
        base_mean_ia = get_mean_ia_idx(n, neighbors)

        self.neighbor_indices = [base_sigma_ii, base_sigma_ia, base_sigma_aa, base_mean_ia]

        print("successfully update neighbor indices")

    def _build_likelihood(self):
        return tf.constant(0, dtype=settings.float_type)

    def get_linear_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="linear_weight")


class GCGP(GCGP_base):

    def __init__(self, X, adj, layers, sample="diagonal", n_samples=1, K=3, neighbors=None, loss_type="link_full",
                 label=None, pos_edges=None, neg_edges=None, idx_train=None, linear_layer=False, **kwargs):

        assert sample in ["diagonal", "full", "neighbor"]
        assert loss_type in ["link_full", "link", "classification", "classification_softmax", "regression"]

        GCGP_base.__init__(self, X, adj, layers, sample, n_samples, K, neighbors, loss_type,
                           label, pos_edges, neg_edges, idx_train, linear_layer, **kwargs)
