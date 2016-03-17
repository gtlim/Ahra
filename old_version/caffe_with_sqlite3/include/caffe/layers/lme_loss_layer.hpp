#ifndef CAFFE_LME_LOSS_LAYER_HPP_
#define CAFFE_LME_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
/** 
 *
 * @brief this is main loss for computing 
 *  Large Margin Embedding 
 *  it is highly optimized 
 **/
template <typename Dtype>
class LmeLossLayer : public LossLayer<Dtype> {
 public:
  explicit LmeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LmeLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }


 protected:
  /// @copydoc CategoryLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
 //This is for cuda implementation
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Dtype margin,thresh,sigma;
  int count;
};

}  // namespace caffe

#endif  // CAFFE_LME_LOSS_LAYER_HPP_
