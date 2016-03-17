#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h> 
#include <memory>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/incremental_data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

#include <sys/stat.h> // for checking.
//#include "caffe/util/db_sqlite3.hpp"

namespace caffe {

template <typename Dtype>
void IncrementalDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  data_transformer_.reset( new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  
  // Read the file with filenames and labels
 
  const int batch_size = this->layer_param_.incremental_data_param().batch_size();

  // Read an image, and use it to initialize the top blob.
  Datum datum;
  db = shared_ptr<db::DB>(db::GetDB(this->layer_param_.incremental_data_param().backend()));
  db->Open(this->layer_param_.incremental_data_param().source(), db::READ);
  cursor = shared_ptr<db::Cursor>(db->NewCursor());
  datum.ParseFromString(cursor->value());

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  batch.data_.Reshape(top_shape);
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  batch.label_.Reshape(label_shape);
  //img_id to label data in loss layer.
  vector<int> id_shape(1, batch_size);
  top[2]->Reshape(id_shape);
  img_id_.Reshape(id_shape);
}


// This function is called on prefetch thread
template <typename Dtype>
void IncrementalDataLayer<Dtype>::LoadData() {
  IncrementalDataParameter incremental_data_param = this->layer_param_.incremental_data_param();
  const int batch_size = incremental_data_param.batch_size();
  // Read an image, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor->value());

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch.data_.Reshape(top_shape);

  Dtype* load_data = batch.data_.mutable_cpu_data();
  Dtype* load_label = batch.label_.mutable_cpu_data();
  Dtype* load_id = img_id_.mutable_cpu_data();

  // datum scales
  //int logger = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    Datum datum;
    datum.ParseFromString(cursor->value());

    // Apply transformations (mirror, crop...) to the image
    int offset = batch.data_.offset(item_id);
    this->transformed_data_.set_cpu_data(load_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    load_label[item_id] = datum.label();
    load_id[item_id] = atoi(cursor->key().c_str());
    // go to the next iter
    cursor->Next();
    if (!cursor->valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor->SeekToFirst();
      // Checking phase;
    }
  }
}

template <typename Dtype>
void IncrementalDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LoadData();
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch.data_);
  // Copy the data
  caffe_copy(batch.data_.count(), batch.data_.cpu_data(),
             top[0]->mutable_cpu_data());
  // Reshape to loaded labels.
  top[1]->ReshapeLike(batch.label_);
  // Copy the labels.
  caffe_copy(batch.label_.count(), batch.label_.cpu_data(),
        top[1]->mutable_cpu_data());
  top[2]->ReshapeLike(img_id_);
  caffe_copy(img_id_.count(), img_id_.cpu_data(),
        top[2]->mutable_cpu_data());
}

INSTANTIATE_CLASS(IncrementalDataLayer);
REGISTER_LAYER_CLASS(IncrementalData);

}  // namespace caffe
#endif  // USE_OPENCV

