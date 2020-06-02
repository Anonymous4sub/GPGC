## dGPGC: Gaussian Process with Graph Convolutional Kernel

This repository contains the tensorflow implementation for GPGC

Further details about GPGC can be found in our paper.

## Requirements
* python 3.6
* Tensorflow 1.14
* GPflow 1.5.1
* scikit-learn
* scipy

## Run the demo

To perform link prediction on Cora dataset, run the following command:

```bash
python link.py --dataset cora --sample neighbor
```