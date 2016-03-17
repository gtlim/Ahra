#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/lstm/Dataset/
DATA=/lstm/Dataset
TOOLS=/v8/ahra/caffe/build/tools

GLOG_logtostderr=1 $TOOLS/compute_image_mean $EXAMPLE/train_awa_icml_25_lmdb \
  $DATA/train_awa_google25.binaryproto

echo "Done."
