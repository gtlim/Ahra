#!/usr/bin/env sh
#--weights=/v9/kakao/snapshot/cnn_hangul_72x72_iter_145000.caffemodel\
#--snapshot=/v9/kakao/snapshot/incremental_inter_iter_51000.solverstate \
/v9/kakao_code/caffe/build/tools/caffe train --gpu=1 \
 --solver=/v9/kakao_code/models/incremental/solver_alex.prototxt \
 --weights=/v9/kakao/snapshot/cnn_hangul_72x72_sec_iter_80000.caffemodel\

 

