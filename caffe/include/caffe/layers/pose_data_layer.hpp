#ifndef CAFFE_POSE_DATA_LAYERS_HPP_
#define CAFFE_POSE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template<typename T1, typename T2, typename T3>
struct triplet
{
    T1 first; 
    T2 second;
    T3 last;
};

template<typename T1, typename T2, typename T3>
triplet<T1,T2,T3> make_triplet(const T1 &m1, const T2 &m2, const T3 &m3) 
{
    triplet<T1,T2,T3> ans;
    ans.first = m1;
    ans.second = m2;
    ans.last = m3;
    return ans;
}

template <typename Dtype>
class Patch {
 public:
  Blob<Dtype> data_, azim_, elev_;
};

template <typename Dtype>
class PoseDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit PoseDataLayer(const LayerParameter& param);
  virtual ~PoseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Patch<Dtype>* batch);
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();

  Patch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Patch<Dtype>*> prefetch_free_;
  BlockingQueue<Patch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_;

  vector<triplet<std::string,float,float> > lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_POSE_DATA_LAYERS_HPP_
