#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define SUM 28
#define MAXDIM 10
#define PROTOS 5

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class spLmeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  spLmeLossLayerTest()
      : blob_bottom_data(new Blob<Dtype>(16, SUM, 1, 1)),
        blob_bottom_score(new Blob<Dtype>(SUM, SUM, 1, 1)),
        blob_bottom_label(new Blob<Dtype>(16, 1, 1, 1)),
        blob_bottom_table(new Blob<Dtype>(1,SUM, 3, 1)),
        blob_bottom_graph(new Blob<Dtype>(1, SUM , MAXDIM , 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FLAGS_logtostderr = 1;
    const int PROTO_NUMS[] = { 5,4,6,10,3};
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data);
    blob_bottom_vec_.push_back(blob_bottom_data);
    filler.Fill(this->blob_bottom_score);
    blob_bottom_vec_.push_back(blob_bottom_score);
    for (int i = 0; i < 8; ++i) {
      blob_bottom_label->mutable_cpu_data()[i] = caffe_rng_rand() % SUM;
      //blob_bottom_label->mutable_cpu_data()[i] = 1;
    }
    for (int i = 8; i < blob_bottom_label->count(); ++i) {
      blob_bottom_label->mutable_cpu_data()[i] = -1;
      //blob_bottom_label->mutable_cpu_data()[i] = caffe_rng_rand() % SUM;
    }
    int offset = 0;
    blob_bottom_vec_.push_back(blob_bottom_label);
    for (int i = 0; i < PROTOS; ++i) { 
     for (int j = 0; j < PROTO_NUMS[i]; ++j) {
       blob_bottom_table->mutable_cpu_data()[(offset+j)*3] = i;
       blob_bottom_table->mutable_cpu_data()[(offset+j)*3+1] = PROTO_NUMS[i];
       blob_bottom_table->mutable_cpu_data()[(offset+j)*3+2] = offset;
     }
     offset+=PROTO_NUMS[i];
    }
    blob_bottom_vec_.push_back(blob_bottom_table);
    /*offset = 0;
    int sp = 0;
    for (int i = 0; i < PROTOS; ++i) { 
     for (int j = 0; j < PROTO_NUMS[i]; ++j) {
      for (int k = 0; k < GRAPH_NUMS[i]; ++k) {
       blob_bottom_graph->mutable_cpu_data()[(sp)*2] = offset+j;
       blob_bottom_graph->mutable_cpu_data()[(sp)*2+1] = caffe_rng_rand() % PROTO_NUMS[i];
       sp++;
      }
     }
     // LOG(INFO) << sp;
     offset+=PROTO_NUMS[i];
    }*/
    for (int i = 0; i < SUM; ++i) { 
     for (int j = 0; j < MAXDIM; ++j) {
       if( j < PROTO_NUMS[i] ) {
        blob_bottom_graph->mutable_cpu_data()[i*MAXDIM+j] = caffe_rng_rand() % 1;
       } else {
        blob_bottom_graph->mutable_cpu_data()[i*MAXDIM+j] = 0;
       }
     }
    }
    blob_bottom_vec_.push_back(blob_bottom_graph);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~spLmeLossLayerTest() {
    delete blob_bottom_data;
    delete blob_bottom_score;
    delete blob_bottom_label;
    delete blob_bottom_table;
    delete blob_bottom_graph;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data;
  Blob<Dtype>* const blob_bottom_score;
  Blob<Dtype>* const blob_bottom_label;
  Blob<Dtype>* const blob_bottom_table;
  Blob<Dtype>* const blob_bottom_graph;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(spLmeLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(spLmeLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<spLmeLossLayer<Dtype> > layer(
        new spLmeLossLayer<Dtype>(layer_param));
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


TYPED_TEST(spLmeLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
  FLAGS_logtostderr = 1;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    spLmeLossParameter* splme_loss_param =
        layer_param.mutable_splme_loss_param();
    //splme_loss_param->set_thresh(0.01);
    splme_loss_param->set_thresh(100);
    splme_loss_param->set_margin2(2);
    spLmeLossLayer<Dtype> layer(layer_param);
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

}// namespace caffe
