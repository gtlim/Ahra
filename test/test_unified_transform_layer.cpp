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
class UnifiedTransformLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  UnifiedTransformLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 10, 1, 1)),
        blob_top_c(new Blob<Dtype>()), 
        blob_top_a(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_a);
    blob_top_vec_.push_back(blob_top_c);
  }
  virtual ~UnifiedTransformLayerTest() {
    delete blob_bottom_;
    delete blob_top_a;
    delete blob_top_c;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_c;
  Blob<Dtype>* const blob_top_a;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UnifiedTransformLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnifiedTransformLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  UnifiedTransformParameter* unified_transform_param =
       layer_param.mutable_unified_transform_param();
  unified_transform_param->set_num_output(5);
  shared_ptr<UnifiedTransformLayer<Dtype> > layer(
      new UnifiedTransformLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_a->num(), 5);
  EXPECT_EQ(this->blob_top_a->height(), 1);
  EXPECT_EQ(this->blob_top_a->width(), 1);
  //changed dimension
  EXPECT_EQ(this->blob_top_a->channels(), 5);
}

TYPED_TEST(UnifiedTransformLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    UnifiedTransformParameter* unified_transform_param =
        layer_param.mutable_unified_transform_param();
    unified_transform_param->set_num_output(5);
    unified_transform_param->mutable_weight_filler()->set_type("uniform");
    shared_ptr<UnifiedTransformLayer<Dtype> > layer(
        new UnifiedTransformLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_a->cpu_data();
    const int count = this->blob_top_a->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
    const Dtype* data2 = this->blob_top_c->cpu_data();
    const int count2 = this->blob_top_c->count();
    for (int i = 0; i < count2; ++i) {
      EXPECT_GE(data2[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(UnifiedTransformLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    UnifiedTransformParameter* unified_transform_param =
        layer_param.mutable_unified_transform_param();
    unified_transform_param->set_num_output(10);
    unified_transform_param->mutable_weight_filler()->set_type("gaussian");
    UnifiedTransformLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-1, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
