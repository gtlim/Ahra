/**
 * @brief linear transform layer
 *   which perform linear transform for 
 *   Devise Ranking Loss 
 *   It can be use for wasabile as well.
 */
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/linear_transform_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LinearTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.linear_transform_param().num_output();
  //When you linearly transform Word Data to Some Semantic Space
  backprop = this->layer_param_.linear_transform_param().backprop();
  //axis should be always 1 
  const int axis = bottom[0]->CanonicalAxisIndex(
        this->layer_param_.linear_transform_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed. -> N x [CHW]
  K_ = bottom[0]->count(axis);  // C * H * W = C, because H & W is 1.
  N_ = num_output;              // Dimension of Joint_Space or Word_Space 
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else {
    //Initialize the weight 2 dimensional matrix
    vector<int> weight_shape(2);
    weight_shape[0] = N_; //          [ N_ x  K_ ] -> W 
    weight_shape[1] = K_; //          it will be transformed, when we use blas.
    //allocate memory to weight matrix 
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.linear_transform_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // parameter initialization
   }
   this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LinearTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  // axis should be always 1
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.linear_transform_param().axis());
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);   //axis is 1, therefore axis + 1 = 2.
  top_shape[axis] = N_;         //N_ is num_output
  top[0]->Reshape(top_shape);
  //display only at train Phase 
  if(this->layer_param_.linear_transform_param().verbose() && this->phase_ == 0) {
    LOG(INFO) <<"||W||:" << this->blobs_[0]->sumsq_data();
  }
}

template <typename Dtype>
void LinearTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // Trans weight matrix
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
}

template <typename Dtype>
void LinearTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }
  // Default backprop = 1;
  if (propagate_down[0] && backprop) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(LinearTransformLayer);
#endif

INSTANTIATE_CLASS(LinearTransformLayer);
REGISTER_LAYER_CLASS(LinearTransform);

}  // namespace caffe
