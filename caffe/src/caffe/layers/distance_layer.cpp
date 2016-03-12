/*
 *
 * @brief this is for generic verison of 
 *  computing distance between instance and proto types
 *  in case of CEDL, there is two types of proto
 *  one is for category proto 
 *   and the other is for attribute 
 *  good luck implemented by gtlim 2015.9.24
 *
 *  please synchronize cpp and cu version at the same time.
 *  this layer is highly optimizied by using cuda implementation
 * 
 */
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/distance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
 
template <typename Dtype>
void DistanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

   //Only 2D -matrix possible
   CHECK_EQ(bottom[0]->channels(),bottom[1]->channels()) <<" Dimension is not compatible ";  
   //Figure out the dimensions. 
   inner_num_ = N_ = bottom[1]->num();	 // number of proto_types 
   inner_dim_ = K_ = (bottom[1]->count())/N_; // dimension of proto
   outter_num_= M_ =  bottom[0]->num();        // batch_size 
   //output is result of computation of similarity. 
   vector<int> top_shape = bottom[0]->shape();
   top_shape.resize(2);
   top_shape[1] = inner_num_;
   top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DistanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> shapen(1);
  shapen[0] = inner_dim_;
  const Dtype* img_feat = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* proto = bottom[1]->cpu_data();
  //computing Distance between proto and img_feat
  Blob<Dtype> sub(shapen); 
  for(int o = 0; o < outter_num_ ;o++) {
   const Dtype* curr_img = &img_feat[o*inner_dim_];
   for(int i = 0; i < inner_num_ ;i++) {
    const Dtype* curr_proto = &proto[i*inner_dim_];
    caffe_sub(inner_dim_,curr_img,curr_proto,sub.mutable_cpu_data());
    top_data[o*inner_num_+i] = sub.sumsq_data();
  }
 }
}
template <typename Dtype>
void DistanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 const Dtype* img_feat = bottom[0]->cpu_data();
 const Dtype* proto  = bottom[1]->cpu_data();

 if (propagate_down[0]) {
  const Dtype* top_diff = top[0]->cpu_diff();
  caffe_set(bottom[0]->count(),Dtype(0.),bottom[0]->mutable_cpu_diff());
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  //Implementation of CEDL
  //backprop to img_feat
  for(int o = 0; o < outter_num_ ;o++) {
   const Dtype* curr_img = &img_feat[o*inner_dim_];
   for(int i = 0; i < inner_num_ ;i++) {
    const Dtype* curr_proto = &proto[i*inner_dim_];
    for(int j = 0 ; j < inner_dim_ ;j++) {
     bottom_diff[o*inner_dim_+j] += top_diff[o*inner_num_+i]*(curr_img[j]-curr_proto[j]);
    }
   }
  }
  caffe_scal(bottom[0]->count(), Dtype(2.), bottom_diff);
 }
 
 if (propagate_down[1] ) {
  const Dtype* top_diff = top[0]->cpu_diff();
  //backprop to proto type
  caffe_set(bottom[1]->count(),Dtype(0.),bottom[1]->mutable_cpu_diff());
  Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
  for(int i = 0; i < inner_num_ ;i++) {
   const Dtype* curr_proto = &proto[i*inner_dim_];
   for(int o = 0; o < outter_num_ ; o++) {
    const Dtype* curr_img = &img_feat[o*inner_dim_];
    for(int j = 0 ; j < inner_dim_ ;j++) {
     bottom_diff[i*inner_dim_+j] +=  top_diff[o*inner_num_+i]*(curr_proto[j]-curr_img[j]);
     }
    }
   }
  caffe_scal(bottom[1]->count(), Dtype(2.), bottom_diff);
 }
}
#ifdef CPU_ONLY
STUB_GPU(DistanceLayer);
#endif

INSTANTIATE_CLASS(DistanceLayer);
REGISTER_LAYER_CLASS(Distance);

}  // namespace caffe
