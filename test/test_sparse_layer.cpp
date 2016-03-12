#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {


template <typename TypeParam>
class SparseTestLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SparseTestLayerTest()  
      : blob_bottom_cdata(new Blob<Dtype>(15, 20, 1, 1)),   
        blob_bottom_adata(new Blob<Dtype>(25, 20, 1, 1)), 
        blob_bottom_sdata(new Blob<Dtype>(20, 15, 1, 1)),   
        blob_bottom_ndata(new Blob<Dtype>(20, 20, 1, 1)),   
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_cdata);
    blob_bottom_vec_.push_back(blob_bottom_cdata);
    filler.Fill(this->blob_bottom_adata);
    blob_bottom_vec_.push_back(blob_bottom_adata);
    for (int i = 0; i < blob_bottom_sdata->count(); ++i) {
      blob_bottom_sdata->mutable_cpu_data()[i] = Dtype((caffe_rng_rand() % 100))/100;
    }
    blob_bottom_vec_.push_back(blob_bottom_sdata);
    filler.Fill(this->blob_bottom_ndata);
    blob_bottom_vec_.push_back(blob_bottom_ndata);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SparseTestLayerTest() {
    delete blob_bottom_cdata;
    delete blob_bottom_adata;
    delete blob_bottom_sdata;
    delete blob_bottom_ndata;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_cdata;
  Blob<Dtype>* const blob_bottom_adata;
  Blob<Dtype>* const blob_bottom_sdata;
  Blob<Dtype>* const blob_bottom_ndata; 
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SparseTestLayerTest, TestDtypesAndDevices);

TYPED_TEST(SparseTestLayerTest, TestForwardCPU) {
  typedef typename TypeParam::Dtype Dtype;
  FLAGS_logtostderr = 1;
  if (Caffe::mode() == Caffe::CPU ) {
    LayerParameter layer_param;
    SparseTestParameter* sparse_test_param = layer_param.mutable_sparse_test_param();
    sparse_test_param->set_dim(20);
    sparse_test_param->set_sigma(0.5);
    sparse_test_param->set_lambda(1.0);
    shared_ptr<SparseTestLayer<Dtype> > layer(
        new SparseTestLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    //const Dtype* data = this->blob_top_->cpu_data();
    //const int count = this->blob_top_->count();
    //for (int i = 0; i < count; ++i) {
    //  EXPECT_GE(data[i], 1.);
  }
}


}// namespace caffe
