
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multi_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define DIM 3

namespace caffe {

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.multi_accuracy_param().top_k();
  LOG(INFO) << "TOP_K : " << top_k_;
  ver_ = this->layer_param_.multi_accuracy_param().version();
  // to evaluate analogical transfer.
  filename = this->layer_param_.multi_accuracy_param().filename();
  if( filename.length() > 0 ) {
    LOG(INFO) << "Filename : " << filename;
    save = true;
  }
}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  multi_num  = bottom[2]->num();
  multi_col  = bottom[2]->count()/multi_num;
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::fstream infile;
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* bottom_mlabel = bottom[2]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  int count = 0;
  if(save) infile.open(filename.c_str(),std::ios::out|std::ios::app);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      //label_index => true label
      const int label_index =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      DCHECK_GE(label_index, 0);
      DCHECK_LT(label_index, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      int top_k;
      if( top_k_ == 1) {
       top_k = top_k_;
      } else { 
       top_k = num_labels;
      }
      if( ver_ == "distance") {
       std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k,
          bottom_data_vector.end(), std::less<std::pair<Dtype, int> >());
      } else if ( ver_ == "similarity" ) { 
       std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());   
      } else {
        LOG(FATAL) << this->type() 
           <<" Layer does not have " << ver_ << " version";
      }
     // check if true label is in top k predictions
      bool checker = false;
      vector<int> top_ks;
      if( top_k_ > 1) {
        int total = 0,i = 0;
        int pre_val = -1;
        while(total < top_k_ ) {
         const int label_value = static_cast<int>(bottom_mlabel[bottom_data_vector[i].second*DIM]);
         if( pre_val != label_value) { 
          total++;
          top_ks.push_back(i);
          pre_val = label_value;
         }
         i++;
        }

      } else {
       top_ks.push_back(0);
      }
      for (int k = 0; k < top_k_; k++) {
       j = top_ks[k];
       const int label_value = static_cast<int>(bottom_mlabel[bottom_data_vector[j].second*DIM]);
       if ( label_index == label_value) { 
          ++accuracy;
          checker = true;
          break;
       }
      } 
      if(save) { 
       int val = ( checker ) ? 1:0;
       infile << label_index << " " << bottom_data_vector[0].first << " " 
          << bottom_data_vector[0].second  << " " << val << std::endl;
     }
     ++count;
   }
  }
  if(save) infile.close();
  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MultiAccuracyLayer);
REGISTER_LAYER_CLASS(MultiAccuracy);

}  // namespace caffe
