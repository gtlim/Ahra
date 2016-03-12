#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/pose_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h>       /* sqrt */

#define THRESH 0.5236
#define PI 3.14159265
namespace caffe {

template <typename Dtype>
float compute_distance(const Dtype azi, const Dtype elev,const Dtype azi2,const Dtype elev2) {
 
 float X1 = cos(azi * PI / 180.0) * sin( elev * PI/180.0);
 float Y1 = sin(elev * PI / 180.0) * sin( azi * PI/180.0);
 float Z1 = cos(elev * PI /180.0);
 
 float X2 = cos(azi2 * PI / 180.0) * sin( elev2 * PI/180.0);
 float Y2 = sin(elev2 * PI / 180.0) * sin( azi2* PI/180.0);
 float Z2 = cos(elev2 * PI /180.0);
  
 return  sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2) + (Z1-Z2)*(Z1-Z2));
}

template <typename Dtype>
void PoseAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.pose_accuracy_param().top_k();
  mode_ = this->layer_param_.pose_accuracy_param().mode();
  version = this->layer_param_.pose_accuracy_param().version();
}

template <typename Dtype>
void PoseAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.pose_accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

#define DIM 2
template <typename Dtype>
void PoseAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::fstream infile;
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_azimuth = bottom[1]->cpu_data();
  const Dtype* bottom_elevation = bottom[2]->cpu_data();
  const Dtype* poseData = bottom[3]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      // Top-k accuracy
      const int azimuth =
          static_cast<int>(bottom_azimuth[i * inner_num_ + j]);
      const int elevation =
          static_cast<int>(bottom_elevation[i * inner_num_ + j]);
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }

      if( mode_ == "distance" ) {
       std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::less<std::pair<Dtype, int> >());
      } else if( mode_ == "similarity") { 
       std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      } else {
       LOG(WARNING) << "wrong version " << mode_;
      }
 
      // check if true label is in top k predictions
      if( version == "thresh") {
       for (int k = 0; k < top_k_; k++) {
         int pos = bottom_data_vector[k].second;
         int azi = poseData[pos*DIM];
         int ele = poseData[pos*DIM+1];
         Dtype result = compute_distance(azi,ele+90,azimuth,elevation+90);
         if ( result <= THRESH ) {
           ++accuracy;
           break;
         }
       }
      } else if( version == "exact") {
       for (int k = 0; k < top_k_; k++) {
         int pos = bottom_data_vector[k].second;
         int azi = poseData[pos*DIM];
         int ele = poseData[pos*DIM+1];
         //LOG(INFO) << pos << " " << azi << " " << ele << " : " << azimuth << " " << elevation;
         if ( azi == azimuth && ele == elevation ) {
           ++accuracy;
           break;
         }
       }
      } else {
       LOG(WARNING) << "Wrong version in pose accuracy layer: " << version;
      }
      ++count;
    }
  }
  //LOG(INFO) <<"Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(PoseAccuracyLayer);
REGISTER_LAYER_CLASS(PoseAccuracy);

}  // namespace caffe
