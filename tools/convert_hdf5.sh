#!/usr/bin/env sh
TOOLS=/v8/ahra/caffe/build/tools
DATA_ROOT=
ROOT=/home/gtlim/project/word/
#LAYERNAME=protos
LAYERNAME=protos
echo "Creating hdf5 file..."

GLOG_logtostderr=1 $TOOLS/convert_hdf5 \
    --weights=$DATA_ROOT/splme_pascal3d3__iter_1000.caffemodel\
    --layer=$LAYERNAME \
    $ROOT \
    /lstm/pascal3d2\

echo "Done."
