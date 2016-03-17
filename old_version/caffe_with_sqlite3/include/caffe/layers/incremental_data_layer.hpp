#ifndef CAFFE_INCREMENTAL_DATA_LAYER_HPP_
#define CAFFE_INCREMENTAL_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/db_sqlite3.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class IncrementalDataLayer : public Layer<Dtype> {
 public:
  explicit IncrementalDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param),transform_param_(param.transform_param()) {}
  virtual ~IncrementalDataLayer() {} 
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "IncrementalData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  // label is provided as -1 when it's unlabeled data(new image file)
  virtual inline int ExactNumTopBlobs() const { return 3; }
 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void LoadData();

  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  Blob<Dtype> transformed_data_;
  Batch<Dtype> batch;
  Blob<Dtype> img_id_;

  vector<int> lines_;
  SQLITECursor* DB_;
  int lines_id_;
  int db_id_;
};

} // namespace caffe

#endif // CAFFE_INCREMENTAL_DATA_LAYER_HPP_
