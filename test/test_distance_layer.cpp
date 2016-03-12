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

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif


template <typename TypeParam>
class DistanceLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  DistanceLayerTest()
      : blob_bottom_vdata(new Blob<Dtype>(10, 30, 1, 1)),
        blob_bottom_wdata(new Blob<Dtype>(20, 30, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_vdata);
    blob_bottom_vec_.push_back(blob_bottom_vdata);
    filler.Fill(this->blob_bottom_wdata);
    blob_bottom_vec_.push_back(blob_bottom_wdata);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DistanceLayerTest() {
    delete blob_bottom_vdata;
    delete blob_bottom_wdata;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_vdata;
  Blob<Dtype>* const blob_bottom_wdata;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DistanceLayerTest, TestDtypesAndDevices);

TYPED_TEST(DistanceLayerTest, TestForward) {

  typedef typename TypeParam::Dtype Dtype;
  FLAGS_logtostderr = 1;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<DistanceLayer<Dtype> > layer(
        new DistanceLayer<Dtype>(layer_param));
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


TYPED_TEST(DistanceLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  FLAGS_logtostderr = 1;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    DistanceLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-1,1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

} //namespace caffe
