#include <vector>

#include "caffe/layers/incremental_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void IncrementalDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LoadData();
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch.data_);
  // Copy the data
  caffe_copy(batch.data_.count(), batch.data_.gpu_data(),
             top[0]->mutable_gpu_data());
  // Reshape to loaded labels.
  top[1]->ReshapeLike(batch.label_);
  // Copy the labels.
  caffe_copy(batch.label_.count(), batch.label_.gpu_data(),
        top[1]->mutable_gpu_data());
  // Reshape to loaded ids.
  top[2]->ReshapeLike(img_id_);
  // Copy the ids.
  caffe_copy(img_id_.count(), img_id_.gpu_data(),
        top[2]->mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FORWARD(IncrementalDataLayer);

}  // namespace caffe
