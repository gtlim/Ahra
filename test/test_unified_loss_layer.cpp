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
class UnifiedLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  UnifiedLossLayerTest()
      : blob_bottom_cdata(new Blob<Dtype>(10, 20, 1, 1)),
        blob_bottom_adata(new Blob<Dtype>(10, 30, 1, 1)),
        blob_bottom_clabel(new Blob<Dtype>(10, 1, 1, 1)),
        blob_bottom_aindx(new Blob<Dtype>(20, 30, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_cdata);
    blob_bottom_vec_.push_back(blob_bottom_cdata);
    filler.Fill(this->blob_bottom_adata);
    blob_bottom_vec_.push_back(blob_bottom_adata);
    for (int i = 0; i < blob_bottom_clabel->count(); ++i) {
      blob_bottom_clabel->mutable_cpu_data()[i] = caffe_rng_rand() % 20;
    }
    blob_bottom_vec_.push_back(blob_bottom_clabel);
    for (int i = 0; i < blob_bottom_aindx->count(); ++i) {
      blob_bottom_aindx->mutable_cpu_data()[i] = caffe_rng_rand() % 1;
    }
    blob_bottom_vec_.push_back(blob_bottom_aindx);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UnifiedLossLayerTest() {
    delete blob_bottom_cdata;
    delete blob_bottom_adata;
    delete blob_bottom_clabel;
    delete blob_bottom_aindx;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_cdata;
  Blob<Dtype>* const blob_bottom_adata;
  Blob<Dtype>* const blob_bottom_clabel;
  Blob<Dtype>* const blob_bottom_aindx;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UnifiedLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnifiedLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    UnifiedLossParameter* unified_loss_param =
        layer_param.mutable_unified_loss_param();
    unified_loss_param->set_version("unaryD");
    shared_ptr<UnifiedLossLayer<Dtype> > layer(
        new UnifiedLossLayer<Dtype>(layer_param));
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


TYPED_TEST(UnifiedLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    UnifiedLossParameter* unified_loss_param =
        layer_param.mutable_unified_loss_param();
    unified_loss_param->set_version("unaryD");
    UnifiedLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-1,1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}// namespace caffe
