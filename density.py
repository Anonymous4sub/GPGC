# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:21:10 2020

@author: fangjy
"""

from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
import gpflow
from gpflow import settings
from gpflow.kernels import RBF, Polynomial, Matern32, Matern12, Matern52

from Datasets import SmallGraph
from gcgp.GCGP import GCGP, get_kernels, get_graphconvolutionkernels, init_layers
from utils.util import link_negative_log_likelihood


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', 'data', 'path of datasets')
flags.DEFINE_string('dataset', 'adjnoun', 'Dataset string.')  # 'Karate', 'dolphins', 'lesmis', 'polbooks', adjnoun'
flags.DEFINE_integer("K_hop", 1, "hops of neighbors")
flags.DEFINE_string("layers_dim", '[8]',
                    'number of hidden units of each layer, the last one is the embedding dimension')
flags.DEFINE_string("num_inducing", '[64]', 'number of inducing points of each layer')
flags.DEFINE_string("gc_kernel", "True", "whether to use SparseGraphConvolution kernel")
flags.DEFINE_string("base_kernels", "[Matern32]", "base kernel function")
flags.DEFINE_string("gc_weight", "[True]", "whether to use nonlinear transformation in Graph Convolution kernels")
flags.DEFINE_string('kernel_dim', '[32]', 'dimension of the base kernel, only used when gc_weight is True')
flags.DEFINE_string("mean_function", "linear", "mean function of each layers: zero or linear")
flags.DEFINE_string('white', 'False', 'whether to use whiten representation of inducing points, i.e., set '
                                      'prior of inducing points u to be N(0, I)')
flags.DEFINE_string("sample", "neighbor", "sampling methods")  # diagonal, full, neighbor
flags.DEFINE_integer("n_samples", 1, "number of samples")
flags.DEFINE_integer("n_neighbors", 5, "number of neighbors to be considered in neighbor sampling")
flags.DEFINE_integer("steps", 2000, "steps of optimization")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_integer("eval_samples", 10, "number of samples for evaluation")
flags.DEFINE_string("exp_name", "default_experiment", "experiment name")
flags.DEFINE_string("save_feature", "True", " ")

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s]: %(message)s')

# create file handler which logs even debug messages
if not os.path.exists('log/{}'.format(FLAGS.dataset)):
    if not os.path.exists('log'):
        os.mkdir('log')
    os.mkdir('log/{}'.format(FLAGS.dataset))
fh = logging.FileHandler('log/{}/{}.log'.format(FLAGS.dataset, FLAGS.exp_name))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

tf.logging.set_verbosity(tf.logging.INFO)

# parameter config
layers_dim = eval(FLAGS.layers_dim)
gc_kernel = eval(FLAGS.gc_kernel)

base_kernels = eval(FLAGS.base_kernels)
if len(base_kernels) != len(layers_dim):
    assert len(base_kernels) == 1  # number of base kernels must be consistent with number of layers or 1
    base_kernels = base_kernels * len(layers_dim)  # kernels:[RBF, RBF, RBF,···]

kernel_dim = eval(FLAGS.kernel_dim)
assert len(kernel_dim) == len(base_kernels)  # length of kernel_dim must be consistent with base_kernels

num_inducing = eval(FLAGS.num_inducing)
if len(num_inducing) != len(layers_dim):
    assert len(num_inducing) == 1  # len of num_inducing list must be consistent with len of layers or 1
    num_inducing = num_inducing * len(layers_dim)

gc_weight = eval(FLAGS.gc_weight)
if len(gc_weight) != len(layers_dim):
    assert len(gc_weight) == 1
    gc_weight = gc_weight * len(layers_dim)

white = eval(FLAGS.white)
save_feature = eval(FLAGS.save_feature)


# log parameter settings
def log_parameter_settings():
    tf.logging.info("==========Parameter Settings==========")
    tf.logging.info("dataset: {}".format(FLAGS.dataset))
    tf.logging.info("K_hop: {}".format(FLAGS.K_hop))
    tf.logging.info("layer_dim: {}".format(layers_dim))
    tf.logging.info("num_inducing: {}".format(num_inducing))
    tf.logging.info("gc_kernel: {}".format(gc_kernel))
    tf.logging.info("base_kernels: {}".format(FLAGS.base_kernels))
    tf.logging.info("kernel_dim: {}".format(FLAGS.kernel_dim))
    tf.logging.info("gc_weight: {}".format(gc_weight))
    tf.logging.info("mean_function: {}".format(FLAGS.mean_function))
    tf.logging.info("white: {}".format(white))
    tf.logging.info("sample: {}".format(FLAGS.sample))
    tf.logging.info("n_samples: {}".format(FLAGS.n_samples))
    tf.logging.info("n_neighbors: {}".format(FLAGS.n_neighbors))
    tf.logging.info("steps:{}".format(FLAGS.steps))
    tf.logging.info("lr: {}".format(FLAGS.lr))
    tf.logging.info("eval_samples: {}".format(FLAGS.eval_samples))
    tf.logging.info("save_feature: {}".format(save_feature))
    tf.logging.info("======================================")


def main():

    graph = SmallGraph(FLAGS.dataset, FLAGS.data_path, K_hop=FLAGS.K_hop)
    num_nodes, num_features = graph.feature.shape
    tf.logging.info("dataset:{}, num nodes:{}, num features:{}".format(FLAGS.dataset, num_nodes, num_features))

    # prepare data
    X = tf.placeholder(dtype=settings.float_type, shape=[num_nodes, num_features])
    adj = tf.sparse_placeholder(dtype=settings.float_type, shape=[num_nodes, num_nodes])
    pos_edges = tf.placeholder(dtype=tf.int32, shape=[None, 2])
    neg_edges = tf.placeholder(dtype=tf.int32, shape=[None, 2])
    # label = tf.placeholder(dtype=settings.float_type, shape=[num_nodes, num_nodes])

    input_dim = num_features

    # get kernels and initialize layers
    n_layers = len(layers_dim)
    all_layers_dim = [input_dim] + layers_dim
    if gc_kernel:
        kernels = get_graphconvolutionkernels(base_kernels, all_layers_dim[:-1], kernel_dim, adj, gc_weight)
    else:
        kernels = get_kernels(base_kernels, all_layers_dim[:-1])
    layers = init_layers(graph.adj, graph.feature, kernels, n_layers, all_layers_dim, num_inducing,
                         gc_kernel=gc_kernel, mean_function=FLAGS.mean_function, white=white)
    tf.logging.info("get kernels and layers")

    # build model
    neighbors = graph.get_K_neighbors(FLAGS.n_neighbors) if FLAGS.sample == "neighbor" else None
    # neighbors = graph.get_K_neighbors_prior(FLAGS.n_neighbors) if FLAGS.sample == "neighbor" else None
    n_split = None

    model = GCGP(X, adj, layers, FLAGS.sample, FLAGS.n_samples, FLAGS.n_neighbors, neighbors=neighbors,
                 loss_type="link", pos_edges=pos_edges, neg_edges=neg_edges, n_split=n_split)
    tf.logging.info("successfully build model")

    # get loss
    loss, nlk, kl = model.calculate_loss()
    Fs, Fmeans, Fvars = model.get_forward_samples(FLAGS.eval_samples)  # [ [S, N, D], ]

    # build optimizer
    with tf.variable_scope('adam'):
        opt_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    tf.logging.info("get loss and optimizer")

    # variables to be initialized
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adam')
    if gc_kernel:
        for layer in model.layers:
            tf_vars += layer.kernel.get_convolution_vars()

    # Initialise
    gpu_config = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_config)
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = off
    sess = tf.Session(config=config)
    sess.run(tf.variables_initializer(var_list=tf_vars))
    model.initialize(session=sess)
    tf.logging.info("successfully initialize session")

    # get adj feed dict value
    adj_value = tf.SparseTensorValue(graph.adj[0], graph.adj[1], graph.adj[2])
    feed_dict = {X: graph.feature, adj: adj_value, pos_edges: graph.train_edges, neg_edges: None}
    n_pos_edges = len(graph.train_edges)

    all_false_edges = graph.get_all_false_edges()
    idx = np.arange(len(all_false_edges))
    tf.logging.info("get all false edges")

    # train
    for i in range(FLAGS.steps):

        t = time.time()

        false_edges_idx = np.random.choice(idx, int(n_pos_edges * 2))
        false_edges_value = all_false_edges[false_edges_idx]

        feed_dict.update({neg_edges: false_edges_value})

        _, loss_value, nlk_value, kl_value = sess.run([opt_step, loss, nlk, kl], feed_dict=feed_dict)
        tf.logging.info("Step {}: loss:{:.6f}, nlk:{:.6f}, kl:{:.6f}, "
                        "time:{:.6f}".format(i, loss_value, nlk_value, kl_value, time.time()-t))

        if i % 10 == 0:
            Fs_value, Fmeans_value = sess.run([Fs, Fmeans], feed_dict=feed_dict)
            F, Fmean = Fs_value[-1], Fmeans_value[-1]  # [S, N, D]

            nll_sample, nll_mean = link_negative_log_likelihood(F, Fmean, graph.val_edges)
            # print(nll_sample, nll_mean)
            tf.logging.info("val_nnl_sample: {:.6f}, val_nnl_mean: {:.6f}".format(nll_sample, nll_mean))
            nll_sample, nll_mean = link_negative_log_likelihood(F, Fmean, graph.test_edges)
            tf.logging.info("test_nnl_sample: {:.6f}, test_nnl_mean: {:.6f}".format(nll_sample, nll_mean))

    # evaluate the model
    Fs_value, Fmeans_value = sess.run([Fs, Fmeans], feed_dict=feed_dict)
    F, Fmean = Fs_value[-1], Fmeans_value[-1]  # [S, N, D]

    nll_sample, nll_mean = link_negative_log_likelihood(F, Fmean, graph.val_edges)
    tf.logging.info("val_nnl_sample: {:.6f}, val_nnl_mean: {:.6f}".format(nll_sample, nll_mean))
    nll_sample, nll_mean = link_negative_log_likelihood(F, Fmean, graph.test_edges)
    tf.logging.info("test_nnl_sample: {:.6f}, test_nnl_mean: {:.6f}".format(nll_sample, nll_mean))

    # save transform features
    if save_feature:

        orig_fea = graph.feature
        transformed_fea, Fs_value = sess.run([kernels[0].convolution(orig_fea), Fs], feed_dict=feed_dict)
        F = np.mean(Fs_value[-1], axis=0)

        if not os.path.exists("log/{}/{}".format(FLAGS.dataset, FLAGS.exp_name)):
            os.mkdir("log/{}/{}".format(FLAGS.dataset, FLAGS.exp_name))

        np.save("log/{}/{}/orig_feature.npy".format(FLAGS.dataset, FLAGS.exp_name), orig_fea)
        np.save("log/{}/{}/transformed_feature.npy".format(FLAGS.dataset, FLAGS.exp_name), transformed_fea)
        np.save("log/{}/{}/F.npy".format(FLAGS.dataset, FLAGS.exp_name), F)
        tf.logging.info("successfully saved data!")


if __name__ == "__main__":

    log_parameter_settings()  # log parameter settings
    main()