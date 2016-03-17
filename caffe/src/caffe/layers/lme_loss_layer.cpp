/**
 * @brief 
 *  large maring embedding loss layer.
 *  implemented by gtlim 2015.9.25
 **/

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lme_loss_layer.hpp"

namespace caffe { 

template <typename Dtype>
void predictData(const Dtype* prob_data,const Dtype* label, Dtype* new_label,
   const vector<std::pair<string,int> >& fid,const int outer_num, const int inner_num, const int dim,const Dtype thresh) {

 for( int i = 0; i < outer_num ; i++ ) {
  for( int j = 0; j < inner_num ; j++ ) {
   std::vector<std::pair<Dtype,int> > data_vector;
   const int label_value = static_cast<int>(label[i*inner_num+j]);

   if( label_value != -1 ) {
    new_label[i*inner_num+j] = label_value; 
    continue;
   }

   for( int k = 0 ; k < dim; ++k) {
    data_vector.push_back(std::make_pair(prob_data[ i*dim + k*inner_num + j],k));
   }
   std::partial_sort(data_vector.begin(),data_vector.begin(),
     data_vector.end(),std::less<std::pair<Dtype,int> >());

   if( data_vector[0].first >= thresh ) {
    new_label[i*inner_num+j] = data_vector[0].second;
   } else if( data_vector[0].first < thresh ) {
    new_label[i*inner_num+j] = -1; 
   }
  }
 }
}

template <typename Dtype>
void LmeLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //give margin at the net proto 
  margin = Dtype(this->layer_param_.lme_loss_param().margin());
  //give thresh at the net proto 
  thresh = Dtype(this->layer_param_.lme_loss_param().thresh());
  //sigma  = Dtype(this->layer_param_.lme_loss_param().sigma());
}

template <typename Dtype>
void LmeLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_score = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = Dtype(0);
  int label;
  //computing ranking loss for category.
  for(int i = 0; i < num; ++i) {
     label = static_cast<int>(bottom_label[i]);
     for(int j = 0; j < dim; ++j) {
     //this is pair_wise version of loss 
     if( j == label) {
      Dtype prob = std::max( Dtype(0),
        margin - bottom_score[dim*i + j] );
      if(prob > 0) { 
       loss+=prob;
      }
     } else if ( j != label) { // Wx.'u_c(0) < sigma 
      Dtype prob = std::max( Dtype(0),
        margin + bottom_score[dim*i + j] );
      if(prob > 0) { 
       loss+=prob;
      }
      prob = std::max( Dtype(0),
       margin + bottom_score[i * dim + label] - bottom_score[i*dim + j] );
      if(prob > 0) {
       loss += prob;
      }      
     }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}


template <typename Dtype>
void LmeLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  //Gradient respect to category similarity
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int count = bottom[0]->count();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = top[0]->cpu_diff()[0]/num;
    int label;
    for (int i = 0; i < num; ++i) {
      label = static_cast<int>(bottom_label[i]);
      for(int j = 0; j < dim; ++j) {
       // pair_wise version of loss
       if( j == label) {
        Dtype prob = std::max( Dtype(0),
          margin - bottom_data[dim*i + j] );
        if(prob > 0) { 
          bottom_diff[dim*i + j] = -1;
        }
       } else if ( j != label) { // Wx.'u_c(0) < sigma 
        Dtype prob = std::max( Dtype(0),
          margin  + bottom_data[dim*i + j] );
        if(prob > 0) { 
         bottom_diff[dim*i + j] = 1;
        }
       prob = std::max( Dtype(0),
         margin + bottom_data[i * dim + label] - bottom_data[i*dim + j] );
       if( prob > 0) {
         bottom_diff[i*dim + j] -= 1;
         bottom_diff[i*dim + label] += 1;
       }
      }
    }
   }
   caffe_scal(count, scale, bottom_diff);
  }
}

INSTANTIATE_CLASS(LmeLossLayer);
REGISTER_LAYER_CLASS(LmeLoss);

}  // namespace caffe

