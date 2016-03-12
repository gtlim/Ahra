#ifndef CAFFE_SPLME_LOSS_LAYER_HPP_
#define CAFFE_SPLME_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
/** 
 *
 * @brief this is main loss for computing 
 *    structure preserving large margin embedding which is sp-lme
 *    in this layer we learn multi-prototypes of category 
 *    with structure preserving constraint
 **/
template <typename Dtype>
class spLmeLossLayer : public LossLayer<Dtype> {
 public:
  explicit spLmeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "spLmeLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 5; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    if( bottom_index == 2 || bottom_index == 3 || bottom_index == 4 ) {
      return false;
    } else {
      return true;
   }
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
  Dtype margin,inner_margin,sp_margin,lambda;
  std::vector<std::pair<Dtype,int> > sorter;
  Blob<Dtype> sorter_gpu;
  string version;
  bool verbose;
};

}  // namespace caffe

#endif  // CAFFE_SPLME_LOSS_LAYER_HPP_
