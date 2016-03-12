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
class CategoryLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  CategoryLossLayerTest()
      : blob_bottom_cdata(new Blob<Dtype>(50, 20, 1, 1)),
        blob_bottom_clabel(new Blob<Dtype>(50, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_cdata);
    blob_bottom_vec_.push_back(blob_bottom_cdata);
    for (int i = 0; i < blob_bottom_clabel->count(); ++i) {
      blob_bottom_clabel->mutable_cpu_data()[i] = caffe_rng_rand() % 20;
    }
    blob_bottom_vec_.push_back(blob_bottom_clabel);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CategoryLossLayerTest() {
    delete blob_bottom_cdata;
    delete blob_bottom_clabel;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_cdata;
  Blob<Dtype>* const blob_bottom_clabel;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CategoryLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CategoryLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    CategoryLossParameter* category_loss_param =
        layer_param.mutable_category_loss_param();
    category_loss_param->set_version("pairD");
    shared_ptr<CategoryLossLayer<Dtype> > layer(
        new CategoryLossLayer<Dtype>(layer_param));
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


TYPED_TEST(CategoryLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    CategoryLossParameter* category_loss_param =
        layer_param.mutable_category_loss_param();
    category_loss_param->set_version("pairD");
    CategoryLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-1,1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}// namespace caffe
