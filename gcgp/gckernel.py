# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:22:17 2020

@author: fangjy
"""
import numpy as np
import tensorflow as tf

import gpflow
from gpflow import settings
from gpflow.kernels import Kernel
from gpflow.features import InducingPointsBase


class SparseGraphConvolution(Kernel):
    
    def __init__(self, input_dim, base_kernel, adj, gc_weight=False, out_dim=None, idx=0):
        # self.adj, self.base_kernel
        super().__init__(input_dim, active_dims=None)
        
        assert gc_weight is False or (gc_weight is True and out_dim is not None)
        # super().__init__(out_dim if gc_weight else input_dim)
        
        self.adj = adj  # sparse tensor
        self.gc_weight = gc_weight
        self.base_kernel = base_kernel(out_dim if gc_weight else input_dim, active_dims=None)
        self.idx = idx

        if gc_weight:
            with tf.variable_scope("GraphConvolutionParams_{}".format(str(self.idx))):
                self.W = tf.get_variable(name="gc_w", shape=[input_dim, out_dim], 
                                         dtype=settings.float_type, initializer=tf.glorot_uniform_initializer())
                # self.b = tf.get_variable(name="gc_b", shape=[1, out_dim],
                # dtype=settings.float_type, initializer=tf.glorot_uniform_initializer())

    @gpflow.params_as_tensors
    def convolution(self, X):
        
        result = tf.sparse_tensor_dense_matmul(self.adj, X) 
        
        if self.gc_weight:
            # result = tf.nn.relu(tf.matmul(result, self.W) + self.b)
            result = tf.nn.relu(tf.matmul(result, self.W))
            
        return result
    
    def get_convolution_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GraphConvolutionParams_{}".format(str(self.idx)))
        
    
    @gpflow.params_as_tensors
    def K(self, X, X2=None):
        
        convX = self.convolution(X)
        convX2 = self.convolution(X2) if X2 is not None else X2
        
        return self.base_kernel.K(convX, convX2)

    @gpflow.params_as_tensors
    def K_split(self, X, X2=None, n_split=2000):

        convX = self.convolution(X)
        convX2 = self.convolution(X2) if X2 is not None else X2

        K_list = []
        num = tf.ceil(tf.shape(X)[0] / n_split)

        for i in range(num):
            K_list.append(self.base_kernel.K(convX[i*n_split:(i+1)*n_split], convX2))

        return K_list

    @gpflow.params_as_tensors
    def Kdiag(self, X):
        return self.base_kernel.Kdiag(self.convolution(X))

    def compute_Ku_symmetric(self, Z, jitter=0.0):
        return self.base_kernel.compute_K_symm(Z) + np.eye(Z.shape[0], dtype=settings.float_type) * jitter


class GraphConvolutionInducingpoints(InducingPointsBase):
    pass


"""
class GraphConvolutionInducingpoints(InducingPointsBase):
    
    def __init__(self, Z):
        
        super().__init__(Z)
    
    @gpflow.params_as_tensors
    def Kuu(self, gc_kernel, jitter=settings.jitter):
        return gc_kernel.base_kernel.K(self.Z) + jitter * tf.eye(self.Z.shape[0], dtype=settings.float_type)
    
    @gpflow.params_as_tensors
    def Kuf(self, gc_kernel, Xnew):
        return gc_kernel.base_kernel.K(self.Z, gc_kernel.convolution(Xnew))
    
"""

@gpflow.features.Kuu.register(GraphConvolutionInducingpoints, SparseGraphConvolution)
def Kuu(feat, kern, jitter=settings.jitter):
    func = gpflow.features.Kuu.dispatch(gpflow.features.InducingPoints, gpflow.kernels.Kernel)
    # print("use gc inducing point Kuu with jitter = {}".format(jitter))
    return func(feat, kern.base_kernel, jitter=jitter)


"""
@gpflow.features.Kuu.register(GraphConvolutionInducingpoints, GraphConvolutionKernel)
def Kuu(feat, kern, jitter=settings.jitter):
    #print("original feat:{}".format(feat.Z))
    with gpflow.decors.params_as_tensors_for(feat):
        #print("after type:{}".format(type(feat.Z)))
        return kern.base_kernel.K(feat.Z) + jitter * tf.eye(len(feat), dtype=settings.float_type)
"""


@gpflow.features.Kuf.register(GraphConvolutionInducingpoints, SparseGraphConvolution, object)
def Kuf(feat, kern, Xnew):
    with gpflow.decors.params_as_tensors_for(feat):
        return kern.base_kernel.K(feat.Z, kern.convolution(Xnew))