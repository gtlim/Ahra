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
class ProtoLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ProtoLayerTest()
      : blob_top_c(new Blob<Dtype>()),
        blob_top_a(new Blob<Dtype>()) {
    // fill the values
    blob_top_vec_.push_back(blob_top_c);
    blob_top_vec_.push_back(blob_top_a);
  }
  virtual ~ProtoLayerTest() {
    delete blob_top_c;
    delete blob_top_a;
  }
  Blob<Dtype>* const blob_top_c;
  Blob<Dtype>* const blob_top_a;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ProtoLayerTest, TestDtypesAndDevices);


TYPED_TEST(ProtoLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    ProtoParameter* proto_param =
       layer_param.mutable_proto_param();
    proto_param->set_num_categ(6);
    proto_param->set_num_attr(8);
    proto_param->set_dim(10);
    proto_param->mutable_weight_filler()->set_type("uniform");
    shared_ptr<ProtoLayer<Dtype> > layer(
        new ProtoLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_c->cpu_data();
    const int count = this->blob_top_c->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 0.0);
    }
    const Dtype* dataa = this->blob_top_a->cpu_data();
    const int counta = this->blob_top_a->count();
    for (int i = 0; i < counta; ++i) {
      EXPECT_GE(dataa[i], 0.0);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(ProtoLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    ProtoParameter* proto_param =
       layer_param.mutable_proto_param();
    proto_param->set_num_categ(6);
    proto_param->set_num_attr(8);
    proto_param->set_dim(10);
    proto_param->mutable_weight_filler()->set_type("gaussian");
    ProtoLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-1, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_ );
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


} //namespace caffe

