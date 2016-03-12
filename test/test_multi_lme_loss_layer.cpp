#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define NUM 10

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class MultiLmeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MultiLmeLossLayerTest()
      : blob_bottom_data(new Blob<Dtype>(15, 50, 1, 1)),
        blob_bottom_label(new Blob<Dtype>(15, 1, 1, 1)),
        //blob_bottom_mlabel(new Blob<Dtype>(5, 10, 1, 1)),
        blob_bottom_mlabel(new Blob<Dtype>(5, 3, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data);
    blob_bottom_vec_.push_back(blob_bottom_data);
    for (int i = 0; i < blob_bottom_label->count(); ++i) {
      //blob_bottom_label->mutable_cpu_data()[i] = caffe_rng_rand() % 50;
      blob_bottom_label->mutable_cpu_data()[i] = 0;
    }

    blob_bottom_vec_.push_back(blob_bottom_label);
    int sp =0;
    int ep = NUM;
    int idx = 0;
    for (int i = 0; i < blob_bottom_mlabel->num()-2; ++i) {
     blob_bottom_mlabel->mutable_cpu_data()[i*3] = NUM;
     blob_bottom_mlabel->mutable_cpu_data()[i*3+1] = sp;
     blob_bottom_mlabel->mutable_cpu_data()[i*3+2] = ep;
     sp = ep;
     ep+=NUM; 
     idx++;
    }
    blob_bottom_mlabel->mutable_cpu_data()[idx*3] = 5;
    blob_bottom_mlabel->mutable_cpu_data()[idx*3+1] = sp;
    blob_bottom_mlabel->mutable_cpu_data()[idx*3+2] = sp+5;
    blob_bottom_mlabel->mutable_cpu_data()[++idx*3] = 15;
    blob_bottom_mlabel->mutable_cpu_data()[idx*3+1] = sp+5;
    blob_bottom_mlabel->mutable_cpu_data()[idx*3+2] = sp+5+15;
  
    blob_bottom_vec_.push_back(blob_bottom_mlabel);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MultiLmeLossLayerTest() {
    delete blob_bottom_data;
    delete blob_bottom_label;
    delete blob_bottom_mlabel;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data;
  Blob<Dtype>* const blob_bottom_label;
  Blob<Dtype>* const blob_bottom_mlabel;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiLmeLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultiLmeLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    MultiLmeLossParameter* multi_lme_loss_param =
        layer_param.mutable_multi_lme_loss_param();
    multi_lme_loss_param->set_version("distance");
    shared_ptr<MultiLmeLossLayer<Dtype> > layer(
        new MultiLmeLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


TYPED_TEST(MultiLmeLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
  FLAGS_logtostderr = 1;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    MultiLmeLossParameter* multi_lme_loss_param =
        layer_param.mutable_multi_lme_loss_param();
    multi_lme_loss_param->set_version("distance");
    MultiLmeLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-3, 1e-1,1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}// namespace caffe
