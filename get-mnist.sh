#!/bin/bash

# Define the URLs for the MNIST dataset files from PyTorch mirror
MNIST_URL_TRAIN_IMAGES="https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
MNIST_URL_TRAIN_LABELS="https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
MNIST_URL_TEST_IMAGES="https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
MNIST_URL_TEST_LABELS="https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
 
# Define the download directory
DOWNLOAD_DIR="/mnist"

# Create the download directory if it doesn't exist
mkdir -p $DOWNLOAD_DIR

# Download the files
wget $MNIST_URL_TRAIN_IMAGES -P $DOWNLOAD_DIR
wget $MNIST_URL_TRAIN_LABELS -P $DOWNLOAD_DIR
wget $MNIST_URL_TEST_IMAGES -P $DOWNLOAD_DIR
wget $MNIST_URL_TEST_LABELS -P $DOWNLOAD_DIR

# Unzip the files to root directory
gunzip -c $DOWNLOAD_DIR/train-images-idx3-ubyte.gz > /train-images.idx3-ubyte
gunzip -c $DOWNLOAD_DIR/train-labels-idx1-ubyte.gz > /train-labels.idx1-ubyte
gunzip -c $DOWNLOAD_DIR/t10k-images-idx3-ubyte.gz > /t10k-images.idx3-ubyte
gunzip -c $DOWNLOAD_DIR/t10k-labels-idx1-ubyte.gz > /t10k-labels.idx1-ubyte

# Clean up the downloaded .gz files
rm -rf $DOWNLOAD_DIR

echo "MNIST dataset has been downloaded and unzipped in the root directory."