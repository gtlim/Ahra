#!/usr/bin/env sh
TOOLS=/v8/ahra/caffe/build/tools
TRAIN_DATA_ROOT=/
DATA=/v4/ar_datasets/awa/for_caffe_db/
EXAMPLE=/v8/ahra/datasets/
echo "Creating train sqlite3 file..."

GLOG_logtostderr=1 $TOOLS/convert_sqlite3 \
    $TRAIN_DATA_ROOT \
    $DATA/AWA_Seen40_train.txt\
    $EXAMPLE/AWA_Seen40_train.db \

echo "Done."
