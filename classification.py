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

from Datasets import Graph
from gcgp.GCGP import GCGP, get_kernels, get_graphconvolutionkernels, init_layers
from utils.util import get_classification_label, classification_acc


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', 'data', 'path of datasets')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_integer("K_hop", 2, "hops of neighbors")
flags.DEFINE_string("label_ratio", "None", "ratio of labelled data, default split when label_ratio is None")
flags.DEFINE_string("normalize", "False", "normalize inpute vector to unity")
flags.DEFINE_string("layers_dim", '[32]',
                    'number of hidden units of each layer, the last one is the embedding dimension')
flags.DEFINE_string("num_inducing", '[200]', 'number of inducing points of each layer')
flags.DEFINE_string("gc_kernel", "True", "whether to use SparseGraphConvolution kernel")
flags.DEFINE_string("base_kernels", "[Matern32]", "base kernel function")
flags.DEFINE_string("gc_weight", "[False]", "whether to use nonlinear transformation in Graph Convolution kernels")
flags.DEFINE_string('kernel_dim', '[128]', 'dimension of the base kernel, only used when gc_weight is True')
flags.DEFINE_string("mean_function", "zero", "mean function of each layers: zero or linear")
flags.DEFINE_string('white', 'True', 'whether to use whiten representation of inducing points, i.e., set '
                                     'prior of inducing points u to be N(0, I)')
flags.DEFINE_string("sample", "diagonal", "sampling methods")  # diagonal, full, neighbor
flags.DEFINE_integer("n_samples", 1, "number of samples")
flags.DEFINE_integer("n_neighbors", 5, "number of neighbors to be considered in neighbor sampling")
flags.DEFINE_integer("steps", 10000, "steps of optimization")
flags.DEFINE_float("lr", 0.0001, "learning rate")
flags.DEFINE_integer("eval_samples", 10, "number of samples for evaluation")
flags.DEFINE_float("kl_scale", 0.01, "")
flags.DEFINE_string("softmax", "True", "whether to use softmax loss")
flags.DEFINE_string("q_diag", "False", "")
flags.DEFINE_string("GCGP_X", "False", "")
flags.DEFINE_string("save_feature", "False", " ")
flags.DEFINE_string("linear_layer", "False", "whether to use linear layer after GP layers")
flags.DEFINE_string("exp_name", "default_experiment", "experiment name")

# GCGP_X: mean_function: zero; white: True; lr: 0.01; kl_scale: 0.01; GCGP_X:True
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
label_ratio = eval(FLAGS.label_ratio)
normalize = eval(FLAGS.normalize)
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
softmax = eval(FLAGS.softmax)
q_diag = eval(FLAGS.q_diag)
GCGP_X = eval(FLAGS.GCGP_X)
save_feature = eval(FLAGS.save_feature)
linear_layer = eval(FLAGS.linear_layer)


# log parameter settings
def log_parameter_settings():
    tf.logging.info("==========Parameter Settings==========")
    tf.logging.info("dataset: {}".format(FLAGS.dataset))
    tf.logging.info("K_hop: {}".format(FLAGS.K_hop))
    tf.logging.info("label_ratio: {}".format(label_ratio))
    tf.logging.info("normalize: {}".format(normalize))
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
    tf.logging.info("kl_scale: {}".format(FLAGS.kl_scale))
    tf.logging.info("softmax: {}".format(softmax))
    tf.logging.info("q_diag: {}".format(q_diag))
    tf.logging.info("GCGP_X: {}".format(GCGP_X))
    tf.logging.info("save_feature: {}".format(save_feature))
    tf.logging.info("linear_layer: {}".format(linear_layer))
    tf.logging.info("======================================")


def main():

    graph = Graph(FLAGS.dataset, FLAGS.data_path, exp_type="classification", normalize_feature=normalize,
                  K_hop=FLAGS.K_hop, label_ratio=label_ratio)
    num_nodes, num_features = graph.feature.shape
    tf.logging.info("dataset:{}, num nodes:{}, num features:{}".format(FLAGS.dataset, num_nodes, num_features))

    # prepare data
    X = tf.placeholder(dtype=settings.float_type, shape=[num_nodes, num_features])
    adj = tf.sparse_placeholder(dtype=settings.float_type, shape=[num_nodes, num_nodes])
    y_train, y_val, y_test, idx_train = get_classification_label(graph, GCGP_X)
    if not linear_layer:
        layers_dim[-1] = graph.y_train.shape[1]  # set number of dimension to be number of classes
    if softmax:
        y_train = np.asarray(graph.y_train[graph.train_mask], dtype=settings.float_type)
        if GCGP_X:
            val = np.asarray(graph.y_val[graph.val_mask], dtype=settings.float_type)
            y_train = np.concatenate((y_train, val), axis=0)

    print(y_train.shape)
    print(idx_train.shape)

    input_dim = num_features

    # get kernels and initialize layers
    n_layers = len(layers_dim)
    all_layers_dim = [input_dim] + layers_dim
    if gc_kernel:
        kernels = get_graphconvolutionkernels(base_kernels, all_layers_dim[:-1], kernel_dim, adj, gc_weight)
    else:
        kernels = get_kernels(base_kernels, all_layers_dim[:-1])

    layers = init_layers(graph.adj, graph.feature, kernels, n_layers, all_layers_dim, num_inducing,
                         gc_kernel=gc_kernel, mean_function=FLAGS.mean_function, white=white, q_diag=q_diag)
    tf.logging.info("get kernels and layers")

    # build model
    # neighbors = graph.get_K_neighbors(FLAGS.n_neighbors) if FLAGS.sample == "neighbor" else None
    neighbors = graph.get_K_neighbors_prior(FLAGS.n_neighbors) if FLAGS.sample == "neighbor" else None
    # n_split = 2000 if FLAGS.dataset == "pubmed" else None

    loss_type = "classification_softmax" if softmax else "classification"
    model = GCGP(X, adj, layers, FLAGS.sample, FLAGS.n_samples, FLAGS.n_neighbors,
                 neighbors=neighbors, loss_type=loss_type, label=y_train, idx_train=idx_train,
                 linear_layer=linear_layer)

    tf.logging.info("successfully build model")

    # get loss
    loss, nlk, kl = model.calculate_loss(FLAGS.kl_scale)
    Fs, Fmeans, Fvars = model.get_forward_samples(FLAGS.eval_samples)  # [ [S, N, D], ]
    if linear_layer:
        W = model.W

    # build optimizer
    with tf.variable_scope('adam'):
        opt_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    tf.logging.info("get loss and optimizer")

    # variables to be initialized
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adam')
    if gc_kernel:
        for layer in model.layers:
            tf_vars += layer.kernel.get_convolution_vars()

    if linear_layer:
        tf_vars += model.get_linear_vars()

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
    feed_dict = {X: graph.feature, adj: adj_value}

    # train
    for i in range(FLAGS.steps):

        t = time.time()

        _, loss_value, nlk_value, kl_value = sess.run([opt_step, loss, nlk, kl], feed_dict=feed_dict)
        tf.logging.info("Step {}: loss:{:.6f}, nlk:{:.6f}, kl:{:.6f}, "
                        "time:{:.6f}".format(i, loss_value, nlk_value, kl_value, time.time()-t))

        if i % 10 == 0:
            Fs_value, Fmeans_value = sess.run([Fs, Fmeans], feed_dict=feed_dict)
            F, Fmean = Fs_value[-1], Fmeans_value[-1]  # [S, N, D]

            if linear_layer:
                W_value = sess.run(W, feed_dict=feed_dict)
                F = np.dot(F, W_value)
                Fmean = np.dot(Fmean, W_value)

            acc_sample, acc_mean = classification_acc(F, Fmean, graph.val_mask, y_val)
            tf.logging.info("acc_sample:{:.6}, acc_mean:{:.6f}".format(acc_sample, acc_mean))
            acc_sample, acc_mean = classification_acc(F, Fmean, graph.test_mask, y_test)
            tf.logging.info("Test_acc_sample:{:.6}, Test_acc_mean:{:.6f}".format(acc_sample, acc_mean))

    # evaluate the model
    Fs_value, Fmeans_value = sess.run([Fs, Fmeans], feed_dict=feed_dict)
    F, Fmean = Fs_value[-1], Fmeans_value[-1]  # [S, N, D]

    if linear_layer:
        W_value = sess.run(W, feed_dict=feed_dict)
        F = np.dot(F, W_value)
        Fmean = np.dot(Fmean, W_value)

    acc_sample, acc_mean = classification_acc(F, Fmean, graph.val_mask, y_val)
    tf.logging.info("acc_sample:{:.6}, acc_mean:{:.6f}".format(acc_sample, acc_mean))
    acc_sample, acc_mean = classification_acc(F, Fmean, graph.test_mask, y_test)
    tf.logging.info("Test_acc_sample:{:.6}, Test_acc_mean:{:.6f}".format(acc_sample, acc_mean))

    # save transform features
    if save_feature:

        orig_fea = graph.feature
        transformed_fea, Fmeans_value = sess.run([kernels[0].convolution(orig_fea), Fmeans], feed_dict=feed_dict)
        Fmean = np.mean(Fmeans_value[-1], axis=0)

        if not os.path.exists("log/{}/{}".format(FLAGS.dataset, FLAGS.exp_name)):
            os.mkdir("log/{}/{}".format(FLAGS.dataset, FLAGS.exp_name))

        np.save("log/{}/{}/orig_feature.npy".format(FLAGS.dataset, FLAGS.exp_name), orig_fea)
        np.save("log/{}/{}/transformed_feature.npy".format(FLAGS.dataset, FLAGS.exp_name), transformed_fea)
        np.save("log/{}/{}/Fmean.npy".format(FLAGS.dataset, FLAGS.exp_name), Fmean)
        tf.logging.info("successfully saved data!")


if __name__ == "__main__":

    log_parameter_settings()  # log parameter settings
    main()