#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/lstm/Dataset/
DATA=/lstm/Dataset
TOOLS=/v8/ahra/caffe/build/tools

GLOG_logtostderr=1 $TOOLS/compute_image_mean_sqlite3 \
  --resize_width=256
  --resise_height=256
  $EXAMPLE/ \
  $DATA/.binaryproto

echo "Done."
