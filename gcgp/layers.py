# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:44:36 2020

@author: fangjy
"""

import numpy as np
import tensorflow as tf

import gpflow
from gpflow import settings, transforms
from gpflow.params import Parameter, Parameterized
from gpflow.conditionals import Kuu, Kuf
from gpflow.kernels import RBF, White

from utils.util import *


class svgp_layer(Parameterized):
    
    def __init__(self, kernel, Z, feature, num_outputs, mean_function,
                 gc_kernel=True, white=False, q_diag=False, **kwargs):
        """
        :param kernel:
        :param Z: values of inducing points
        :param feature: inducing points, type: GraphConvolutionInducingpoints
        :param num_outputs:
        :param mean_function:
        :param white: whether set prior of inducing points as N(0, I)
        :param kwargs:
        """
        
        Parameterized.__init__(self, **kwargs)
        # super().__init__(**kwargs)
        
        self.kernel = kernel
        self.num_inducing = Z.shape[0]
        # self.feature = GraphConvolutionInducingpoints(Z)
        self.feature = feature
        
        self.num_outputs = num_outputs
        self.mean_function = mean_function
        self.gc_kernel = gc_kernel
        self.white = white
        self.q_diag = q_diag
        
        self.q_mu, self.q_sqrt = self._init_variational_parameters(Z)
        
        # print(type(self.feature.Z))
        
        self._build_cholesky()

    def _init_variational_parameters(self, Z):
        
        q_mu = np.zeros((self.num_inducing, self.num_outputs))
        q_mu = gpflow.Param(q_mu)

        # initialize q_sqrt to prior
        """
        if not self.white:
            if self.gc_kernel:
                Ku = self.kernel.compute_Ku_symmetric(Z, jitter=settings.jitter)
            else:
                Ku = self.kernel.compute_K_symm(Z) + np.eye(Z.shape[0], dtype=settings.float_type) * settings.jitter

            Lu = np.linalg.cholesky(Ku)
            q_sqrt = np.tile(Lu[None, :, :], [self.num_outputs, 1, 1])
        else:
            q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [self.num_outputs, 1, 1])
        
        # q_sqrt = tf.convert_to_tensor(q_sqrt, dtype=settings.float_type)
        transform = transforms.LowerTriangular(self.num_inducing, num_matrices=self.num_outputs)
        # q_sqrt = Parameter(q_sqrt, transform=transform)
        q_sqrt = gpflow.Param(q_sqrt, transform=transform)
        """

        if self.white or self.q_diag:
            q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [self.num_outputs, 1, 1])
        else:
            if self.gc_kernel:
                Ku = self.kernel.compute_Ku_symmetric(Z, jitter=settings.jitter)
            else:
                Ku = self.kernel.compute_K_symm(Z) + np.eye(Z.shape[0], dtype=settings.float_type) * settings.jitter

            Lu = np.linalg.cholesky(Ku)
            q_sqrt = np.tile(Lu[None, :, :], [self.num_outputs, 1, 1])

        if self.q_diag:
            transform = transforms.DiagMatrix(self.num_inducing)
        else:
            transform = transforms.LowerTriangular(self.num_inducing, num_matrices=self.num_outputs)

        q_sqrt = gpflow.Param(q_sqrt, transform=transform)

        return q_mu, q_sqrt

    def _build_cholesky(self):
        
        self.Ku = Kuu(self.feature, self.kernel, jitter=settings.jitter)
        self.Lu = tf.cholesky(self.Ku)
        self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1])
        self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])

        # self.Kuf = Kuf(self.feature, self.kernel, np.random.uniform(size=(2708, 1433)))

    def sample_from_conditional(self, X, full_cov=False):
        # X: (S, N, D)
        # mean: (S, N, num_outputs), var:(S, N, num_output) or (S, N, N, num_output)
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        S, N, D = tf.shape(X)[0], tf.shape(X)[1], self.num_outputs
        
        mean = tf.reshape(mean, (S, N, D))
        if full_cov:
            var = tf.reshape(var, (S, N, N, D))
        else:
            var = tf.reshape(var, (S, N, D))
            
        z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = self.reparameterize(mean, var, z, full_cov=full_cov) # samples : (S, N, D)
        
        return samples, mean, var

    def sample_from_neighbor_conditional(self, X, K, indices, n_split=None):
        """
        X: (S, N, D_input)
        neighbors: dict({})
        """
        
        # mean: (S, N, D_output) # var: (S, N, D_output)
        mean, var = self.conditional_neighbor_SND(X, K, indices, n_split)
        S, N, D = tf.shape(X)[0], tf.shape(X)[1], self.num_outputs
        
        mean = tf.reshape(mean, (S, N, D))
        var = tf.reshape(var, (S, N, D))
        
        z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = self.reparameterize(mean, var, z, full_cov=False)  # samples : (S, N, D)
        
        return samples, var, mean

    @gpflow.params_as_tensors
    def KL(self):
        """
        The KL divergence from the variational distribution to the prior

        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        # if self.white:
        #     return gauss_kl(self.q_mu, self.q_sqrt)
        # else:
        #     return gauss_kl(self.q_mu, self.q_sqrt, self.Ku)

        KL = -0.5 * self.num_outputs * self.num_inducing
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.q_sqrt) ** 2))

        if not self.white:
            KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Lu))) * self.num_outputs
            KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.Lu_tiled, self.q_sqrt, lower=True)))
            Kinv_m = tf.cholesky_solve(self.Lu, self.q_mu)
            KL += 0.5 * tf.reduce_sum(self.q_mu * Kinv_m)
        else:
            KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt))
            KL += 0.5 * tf.reduce_sum(self.q_mu**2)

        return KL

    def conditional_SND(self, X, full_cov=False):
        
        # mean: (S, N, num_outputs), var:(S, N, num_output) or (S, N, N, num_output)
        f = lambda a : self.conditional_ND(a, full_cov=full_cov)
        mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
        return tf.stack(mean), tf.stack(var)

    @gpflow.params_as_tensors
    def conditional_ND(self, X, full_cov=False):
        # X:(N, n_input)
        """
        uf = Kuf(self.feature, self.kernel, X)
        
        A = tf.matrix_triangular_solve(self.Lu, uf, lower=True)
        if not self.white:
            A = tf.matrix_triangular_solve(tf.transpose(self.Lu), A, lower=False)
        
        mean = self.mean_function(X) + tf.matmul(A, self.q_mu, transpose_a=True)
        """
        
        mean, A = self.calculate_mean(X)
        
        A_tiled = tf.tile(A[None, :, :], [self.num_outputs, 1, 1])
        I = tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :]
        
        if self.white:
            SK = -I
        else:
            SK = - self.Ku_tiled
        
        if self.q_sqrt is not None:
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True)
        
        B = tf.matmul(SK, A_tiled)
        
        if full_cov:
            delta_cov = tf.matmul(A_tiled, B, transpose_a=True)
            Kff = self.kernel.K(X)
        else:
            delta_cov = tf.reduce_sum(A_tiled * B, 1)
            Kff = self.kernel.Kdiag(X)
        
        # either (1, num_X) + (num_outputs, num_X) or (1, num_X, num_X) + (num_outputs, num_X, num_X)
        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)
        
        return mean, var

    @gpflow.params_as_tensors
    def calculate_mean(self, X):
        
        uf = Kuf(self.feature, self.kernel, X)
        A = tf.matrix_triangular_solve(self.Lu, uf, lower=True)

        if not self.white:
            A = tf.matrix_triangular_solve(tf.transpose(self.Lu), A, lower=False)
        
        mean = self.mean_function(X) + tf.matmul(A, self.q_mu, transpose_a=True)
        
        return mean, A
    
    def reparameterize(self, mean, var, z, full_cov=False):
        """
        mean : (S, N, D)
        var: (S, N, D) or (S, N, N, D)
        z: (S, N, D)
        
        return sample:(S, N, D)
        """
        if var is None:
            return mean
        
        if full_cov is False:
            return mean + z * (var + settings.jitter) ** 0.5
        else:
            N = tf.shape(mean)[1]
            mean = tf.transpose(mean, [0, 2, 1]) # SND -> SDN
            var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
            I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None, None, :, :] # 11NN
            chol = tf.cholesky(var + I)  # SDNN
            z_SDN1 = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
            f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
            return tf.transpose(f, (0, 2, 1)) # SND

    @gpflow.params_as_tensors
    def conditional_ND_full_split(self, X, n_split=2000):

        # X:(N, n_input)
        mean, A = self.calculate_mean(X)  # mean: (N, num_output); A:(M, N)

        A_tiled = tf.tile(A[None, :, :], [self.num_outputs, 1, 1])  # (num_output, M, N)
        I = tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :]

        if self.white:
            SK = -I
        else:
            SK = - self.Ku_tiled

        if self.q_sqrt is not None:
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True)

        B = tf.matmul(SK, A_tiled)  # (num_output, M, N)

        # splits data to calculate Kff
        # delta_cov = tf.matmul(A_tiled, B, transpose_a=True)
        # Kff = self.kernel.K(X)
        A_tiled_transpose = tf.transpose(A_tiled, (0, 2, 1))

        Kff_all_list = list()
        num = tf.ceil(tf.shape(X)[0] / n_split)

        for i in range(self.num_outputs):

            Kff_list = []
            for j in tf.range(num):
                Kff_list.append(tf.matmul(A_tiled_transpose[i, j * n_split:(j + 1) * n_split], B[i]))

            Kff_single_output = tf.concat(Kff_list, axis=0)
            Kff_all_list.append(Kff_single_output)

        delta_cov = tf.stack(Kff_all_list, axis=0)  # (num_output, N, N)

        Kff_split = self.kernel.K_split(X, n_split=n_split)
        Kff = tf.concat(Kff_split, axis=0)  # (N, N)

        # (1, N, N) + (num_outputs, N, N)
        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)

        return mean, var

    def conditional_neighbor_SND(self, X, K, indices, n_split):
        # X: (S, N, D)
        f = lambda a: self.conditional_neighbor_ND(a, K=K, indices=indices, n_split=n_split)
        mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
        return tf.stack(mean), tf.stack(var)

    @gpflow.params_as_tensors
    def conditional_neighbor_ND(self, X, K, indices, n_split=None):
        """
        X: (N, D_input)
        neighbors: [[neighbor1,...neighborK], ...] shape: (N, K), every node has fix number of neighbors K
        indices:[sigma_ii_idx, sigma_ia_idx, sigma_aa_idx, mean_ia_idx] indices to gather variances
        
        return: mean: (N, num_outputs), var:(N, num_outputs)
        """
        # mean: (N, num_outputs) full_var: (N, N, num_outputs)
        sigma_ii_idx, sigma_ia_idx, sigma_aa_idx, mean_ia_idx = indices[0], indices[1], indices[2], indices[3]

        if n_split is None:
            mean, full_var = self.conditional_ND(X, full_cov=True)
        else:
            mean, full_var = self.conditional_ND_full_split(X, n_split)
        
        N, num_outputs = tf.shape(mean)[0], tf.shape(mean)[1]
        mean = tf.transpose(mean)  # (num_outputs, N)
        full_var = tf.transpose(full_var)  # (num_outputs, num_X, num_X)

        f_ii = lambda x: tf.gather_nd(x, sigma_ii_idx)
        sigma_ii = tf.reshape(tf.map_fn(f_ii, full_var), (num_outputs, N))  # [num_outputs, N]

        # [num_outputs, N, 1, K]
        f_ia = lambda x: tf.gather_nd(x, sigma_ia_idx)
        sigma_ia = tf.reshape(tf.map_fn(f_ia, full_var), (num_outputs, N, 1, K))  # (num_outputs, N, 1, K)
        sigma_ai = tf.transpose(sigma_ia, (0, 1, 3, 2))

        # [num_outputs, N, K, K]
        f_aa = lambda x: tf.gather_nd(x, sigma_aa_idx)
        sigma_aa = tf.reshape(tf.map_fn(f_aa, full_var), (num_outputs, N, K, K))

        # [1, 1, K, K]
        I = settings.jitter * tf.eye(K, dtype=settings.float_type)[None, None, :, :]
        
        chol = tf.cholesky(sigma_aa + I)  # [num_outputs, N, K, K]
        b_T = tf.matrix_triangular_solve(chol, sigma_ai, lower=True)
        b_T = tf.matrix_triangular_solve(tf.transpose(chol, (0, 1, 3, 2)), b_T, lower=False)  # [num_outputs, N, K, 1]
        b = tf.transpose(b_T, (0, 1, 3, 2))  # [num_outputs, N, 1, K]
        
        # calculate mean
        """
        f_mean_ia = lambda x: tf.gather_nd(x, mean_ia_idx)
        mean_ia = tf.reshape(tf.map_fn(f_mean_ia, mean), (num_outputs, N, K, 1))  # [num_outputs, N, K, 1]
        mean_new = tf.squeeze(tf.matmul(b, mean_ia), axis=[-1, -2])  # [num_outputs, N]
        """

        mean_new = mean  # set sample to mean
        
        # calculate variance
        var_new = sigma_ii - tf.squeeze(tf.matmul(sigma_ia, b_T), axis=[-1, -2])  # [num_outputs, N]
        
        return tf.transpose(mean_new), tf.transpose(var_new)


class GraphConvolution(object):

    def __init__(self, input_dim, out_dim, bias=False, act=tf.nn.relu, sparse_inputs=True):

        self.bias = bias
        self.act = act
        self.sparse_inputs = sparse_inputs

        with tf.variable_scope("GC_transform"):
            self.W = tf.get_variable(name="gc_w", shape=[input_dim, out_dim],
                                     dtype=settings.float_type, initializer=tf.glorot_uniform_initializer())
            if self.bias:
                self.b = tf.get_variable(name="gc_b", shape=[1, out_dim],
                                        dtype=settings.float_type, initializer=tf.zeros_initializer())

    def forward(self, adj, X):

        if self.sparse_inputs:
            feature = tf.sparse_tensor_dense_matmul(adj, X)  # adj is sparse tensor
        else:
            feature = tf.matmul(adj, X)
        feature = tf.matmul(feature, self.W)
        if self.bias:
            feature += self.b

        return self.act(feature)

    @staticmethod
    def get_convolution_vars():
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GC_transform")
