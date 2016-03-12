/**
 * @brief 
 *  this is ranking loss with structure preserving LME
 *  with basic multi-prototype clustering added
 *  structure preserving constraint.
 *  implemented by gtlim 2015.9.25
 **/


/**
 * bottom[0] similarity or distance measure. btw instances
 *  [ batch ] x [ num_cate * num_proto ]
 * bottom[1] similarity or distance measure. btw prototypes
 * bottom[2] label of instance
 * bottom[3] multi_label table of given label
 * [category_label] [ num_proto ] 
 * bottom[4] structure preserving constraint graph
 **/

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layers/splme_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

// define for input information table
#define DIM 3   // dimension of table
#define OFF 2   // offset of given prototypes
#define NUMP 1  // number of prototypes
#define INDEX 0 // cateogry label of given prototypes

namespace caffe {

template <typename Dtype>
void spLmeLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 CHECK_EQ(bottom[0]->num(), bottom[2]->num()) 
  <<"The data and label should have same number.";
 vector<int> loss_shape(0); // Loss layers output a scalar; 0 axis.
 top[0]->Reshape(loss_shape);
 // margin of spLME 
 margin = Dtype(this->layer_param_.splme_loss_param().margin());
 inner_margin = Dtype(this->layer_param_.splme_loss_param().inner_margin());
 sp_margin = this->layer_param_.splme_loss_param().sp_margin();
 lambda = this->layer_param_.splme_loss_param().lambda();
 //verbose 
 verbose = this->layer_param_.splme_loss_param().verbose();
 version = this->layer_param_.splme_loss_param().version();
 if( verbose ) {
  top[1]->Reshape(loss_shape);
  top[2]->Reshape(loss_shape);
 }
}

template <typename Dtype>
void spLmeLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* score  = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[2]->cpu_data();
  // Information table.
  const Dtype* table_  = bottom[3]->cpu_data();
  const int outter_loop = bottom[0]->num(); //batch_size;
  const int outter_dim  = bottom[0]->count()/outter_loop; // total number of prototypes
  Dtype closs = Dtype(0);
  //computing ranking loss for category.
  for(int i = 0; i < outter_loop; ++i) {
   const int label = static_cast<int>(bottom_label[i]);
   const int label_index = table_[label*DIM+INDEX]; //category index
   for(int j = 0; j < outter_dim; ++j) {
    if( table_[j*DIM+INDEX] == label_index) continue;
    const Dtype prob = std::max( Dtype(0),
      margin + score[i*outter_dim + label] - score[i*outter_dim + j] );
    closs+= ( prob > 0 ) ? prob : 0;
   }
  }

  // computing structure preserving constraints
  // precomputing minimum of all category
  // input graph dimension 
  // [ All prototype ] x [ Max number of Prototype ]
  sorter.clear();
  const Dtype* graph  = bottom[4]->cpu_data();   // graph e-nearest-neighbor graph.
  const Dtype* s_value = bottom[1]->cpu_data();  // confidence value of all prototypes.
  const int MaxNum = bottom[4]->count()/outter_dim;  // Max number of prototypes
  for(int i = 0 ; i < outter_dim ; i++) {
   const int num_proto = table_[i*DIM+NUMP];
   const int offset = table_[i*DIM+OFF];
   Dtype max_val = s_value[i*outter_dim+offset ]; 
   int max_idx = offset;
   bool init = false;
   for(int k = 0; k < num_proto; k++) {
    //initial value of connected component
    if( graph[i*MaxNum +k ] && !init ) {
     max_val = s_value[i*outter_dim + offset+k]; 
     max_idx = offset+k;
     init = true;
    }
    if( init && graph[i*MaxNum + k] && max_val  < s_value[i*outter_dim + offset + k] ) {
     max_val = s_value[i*outter_dim + offset + k];
     max_idx = offset + k;
    } 
   }
   sorter.push_back(std::make_pair(max_val,max_idx));
  }
   
  CHECK_EQ(sorter.size(),outter_dim) << "Wrong Computation";
  Dtype sloss = Dtype(0); 
  for( int i = 0 ; i < outter_loop ; i++) {
   const int label = static_cast<int>(bottom_label[i]);
   const int label_index = table_[label*DIM+INDEX]; //category index
   for(int j = 0; j < outter_dim ; ++j) {
    int pos = j - int(table_[j*DIM+OFF]); 
    if( table_[j*DIM+INDEX] == label_index && graph[label*MaxNum + pos] ) 
      continue; //except nearest neighbors.
    const Dtype prob = std::max( Dtype(0),
      margin + sorter[label].first - s_value[label*outter_dim + j] );
    sloss+= ( prob > 0 ) ? prob : 0;
   }
  }
  if( verbose ) {
   top[1]->mutable_cpu_data()[0] = closs;
   top[2]->mutable_cpu_data()[0] = sloss;
  }
  closs+= sloss;
  top[0]->mutable_cpu_data()[0] = closs / outter_loop;
}


template <typename Dtype>
void spLmeLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2] || propagate_down[3] || propagate_down[4]) {
   LOG(FATAL) << this->type() 
               << " Layer cannot backpropagate to label inputs.";
  }

  //Gradient respect to category similarity
  if (propagate_down[0]) {
   const Dtype* score  = bottom[0]->cpu_data();
   const Dtype* bottom_label = bottom[2]->cpu_data();
   const Dtype* table_  = bottom[3]->cpu_data();
   const int outter_loop = bottom[0]->num(); //batch_size;
   const int outter_dim  = bottom[0]->count()/outter_loop; // total number of prototypes

   Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
   caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
   const Dtype scale = top[0]->cpu_diff()[0]/ outter_loop;

   //computing ranking loss for category.
   for(int i = 0; i < outter_loop; ++i) {
    const int label = static_cast<int>(bottom_label[i]);
    if( label == -1 ) continue;
    const int label_index = table_[label*DIM+INDEX]; //category index
    for(int j = 0; j < outter_dim; ++j) {
     if( table_[j*DIM+INDEX] == label_index) continue;
     const Dtype prob = std::max( Dtype(0),
       margin + score[i*outter_dim + label] - score[i*outter_dim + j] );
     if( prob > 0 ) {
      bottom_diff[i*outter_dim + label] += 1;
      bottom_diff[i*outter_dim + j] -= 1;
     }
    }
   }
   caffe_scal(bottom[0]->count(), scale, bottom_diff);
  }

  //Gradient with respect to structure preserving constraint.
  if (propagate_down[1]) {
   const Dtype* graph  = bottom[4]->cpu_data();
   const Dtype* s_value  = bottom[1]->cpu_data();
   const Dtype* bottom_label = bottom[2]->cpu_data();
   const Dtype* table_  = bottom[3]->cpu_data(); 
   const int outter_loop = bottom[0]->num(); //batch_size;
   const int outter_dim  = bottom[0]->count()/outter_loop; // total number of prototypes
   const int MaxNum = bottom[4]->count()/outter_dim;  // Max number of prototypes
   Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
   caffe_set(bottom[1]->count(), Dtype(0), bottom_diff);
   const Dtype scale = top[0]->cpu_diff()[0]/ outter_loop;

   for( int i = 0 ; i < outter_loop ; i++) {
    const int label = static_cast<int>(bottom_label[i]);
    const int label_index = table_[label*DIM+INDEX]; //category index
    for(int j = 0; j < outter_dim ; ++j) { 
     int pos = j - int(table_[j*DIM+OFF]); 
     if( table_[j*DIM+INDEX] == label_index && graph[label*MaxNum + pos ] ) 
        continue; //except nearest neighbors.
     const Dtype prob = std::max( Dtype(0),
       margin + sorter[label].first - s_value[label*outter_dim + j] );
     if( prob > 0 ) {
      bottom_diff[ label*outter_dim + sorter[label].second] += 1;
      bottom_diff[ label*outter_dim + j] -= 1;
     }
    }
   }
   caffe_scal(bottom[1]->count(), scale, bottom_diff);
  }
}

INSTANTIATE_CLASS(spLmeLossLayer);
REGISTER_LAYER_CLASS(spLmeLoss);

}  // namespace caffe


  



