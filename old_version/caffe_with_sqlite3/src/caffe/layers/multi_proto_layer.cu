/** 
 *
 * Updating Uc Multi-prototypes of category
 * implemented by gtlim 2015 9.29
 **/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multi_proto_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void MultiProtoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_copy(top[0]->count(),this->blobs_[0]->gpu_data(),top_data);
}


template <typename Dtype>
void MultiProtoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
     // Gradient with respect to category proto types
     Blob<Dtype> NewBlob(this->blobs_[0]->shape());
     caffe_gpu_set(NewBlob.count(), Dtype(0), NewBlob.mutable_gpu_data());
     caffe_gpu_add(top[0]->count(),top[0]->gpu_diff(),this->blobs_[0]->gpu_diff(),NewBlob.mutable_gpu_data());
     caffe_copy(top[0]->count(),NewBlob.gpu_data(),this->blobs_[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiProtoLayer);

}  // namespace caffe
