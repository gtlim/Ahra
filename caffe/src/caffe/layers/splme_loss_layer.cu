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

// define for sorter 
#define DIMS 2   // dimension of sorter

namespace caffe {

//precompute minimum of all category.
template <typename Dtype>
__global__ void preComputeGPU(const int nthreads,
 const Dtype* s_value, const Dtype* table, const Dtype* graph,
 Dtype* sorter, const int MaxNum) {
 const int outter_dim = nthreads;
 CUDA_KERNEL_LOOP(index, nthreads) {
  const int num_proto = table[index*DIM+NUMP];
  const int offset    = table[index*DIM+OFF];
  Dtype max_val = s_value[index*outter_dim+offset ]; 
  int max_idx = offset;
  bool init = false;
  for(int i = 0 ; i < num_proto; i++) {
    //initial value of connected component
    if( graph[index*MaxNum +i ] && !init ) {
     max_val = s_value[index*outter_dim + offset+i]; 
     max_idx = offset+i;
     init = true;
    }
    if( init && graph[index*MaxNum + i] && max_val  < s_value[index*outter_dim + offset + i] ) {
     max_val = s_value[index*outter_dim + offset + i];
     max_idx = offset + i;
    } 
  }
  sorter[index*DIMS] = max_val;
  sorter[index*DIMS+1] = max_idx;
 }
}

//computing structure preserving constraint loss
template <typename Dtype>
__global__ void spConstraintLossForwardGPU(const int nthreads,
    const Dtype* s_val, const Dtype* bottom_label, Dtype* sloss, 
    const Dtype* graph,const Dtype* table ,const Dtype* sorter,
    const int outter_dim,const int MaxNum,const Dtype margin) {
 CUDA_KERNEL_LOOP(index, nthreads) {
  const int n = index / outter_dim; //row of instance.
  //const int label = static_cast<int>(bottom_label[n]);
  const int label = n;
  const int label_index = table[label*DIM+INDEX];  //category index.
  const int position = index%outter_dim;
  const int offset = table[position*DIM+OFF];
  if(table[position*DIM+INDEX] == label_index && graph[label*MaxNum + position - offset] )
    continue; //except nearest neighbor
  const Dtype prob = max( Dtype(0), 
    margin + sorter[label*DIMS] - s_val[label*outter_dim+position]);
  sloss[index] = (prob > 0 ) ? prob : 0;
 }
}

//computing category loss pair-wise Distance
template <typename Dtype>
__global__ void spLmeLossPair_Distance_ForwardGPU(const int nthreads,
    const Dtype* score, const Dtype* label,const Dtype* table,
    Dtype* loss, const int outter_dim,const Dtype margin,const Dtype inner_margin) {
 CUDA_KERNEL_LOOP(index, nthreads) {
  const int n = index / outter_dim; //row of score.
  const int position = index%outter_dim;
  const int label_value = static_cast<int>(label[n]);
  const int label_index = table[label_value*DIM+INDEX]; //category index.
  const int offset = n*outter_dim + label_value;
  if( offset == index ) {  
   continue; 
  } else if( table[position*DIM+INDEX] != label_index) {
  // pair_wise version of loss 
   loss[index] += max( Dtype(0),
      margin + score[offset] - score[index] );
  } else if( table[position*DIM+INDEX] == label_index && inner_margin > 0) {
  // pair_wise for multi_labels 
   loss[index] += max( Dtype(0),
      inner_margin + score[offset] - score[index] );  
  }
 }
}

//computing category loss pair-wise Distance
template <typename Dtype>
__global__ void spLmeLossUnary_Distance_ForwardGPU(const int nthreads,
    const Dtype* score, const Dtype* label,const Dtype* table,
    Dtype* loss, const int outter_dim,const Dtype margin,const Dtype inner_margin) {
 CUDA_KERNEL_LOOP(index, nthreads) {
  const int n = index / outter_dim; //row of score.
  const int position = index%outter_dim;
  const int label_value = static_cast<int>(label[n]);
  const int label_index = table[label_value*DIM+INDEX]; //category index.
  const int offset = n*outter_dim + label_value;
  if( offset == index ) {  
   loss[index] += max( Dtype(0),
      score[index] - margin );
  } else if( table[position*DIM+INDEX] != label_index) {
   loss[index] += max( Dtype(0),
      inner_margin - score[index]);
  } else if( table[position*DIM+INDEX] == label_index && inner_margin > 0) {
   loss[index] += max( Dtype(0),
      inner_margin - score[index]);
  }
 }
}

template <typename Dtype>
void spLmeLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype closs,wloss,sloss;
  wloss = Dtype(0.);
  //computing category loss 
  const Dtype* score = bottom[0]->gpu_data(); //score of L2 norm or similarity
  const Dtype* bottom_label = bottom[2]->gpu_data(); //own label of instance( which is from clustering)
  const Dtype* table_ = bottom[3]->gpu_data();  // information table
  const int outter_loop = bottom[0]->num(); //batch_size
  const int outter_dim = bottom[0]->count()/outter_loop; // total number of prototypes
 
  const int nthreads = bottom[0]->count(); // [ batch_size x number of prototypes ]
  Blob<Dtype> c_loss(bottom[0]->shape());
  Dtype* loss_data = c_loss.mutable_gpu_data();
  //select version of terms.
  //computing main loss ( large margin embedding with multi-prototypes)
  if( version == "pair" ) {
   spLmeLossPair_Distance_ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
       CAFFE_CUDA_NUM_THREADS>>>(nthreads, score , bottom_label, table_, 
        loss_data, outter_dim, margin,inner_margin);  
  } else if( version == "unary") {
   spLmeLossUnary_Distance_ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
       CAFFE_CUDA_NUM_THREADS>>>(nthreads, score , bottom_label, table_, 
        loss_data, outter_dim, margin,inner_margin);  
  } else if( version == "verify" ) {
   caffe_gpu_set(nthreads,Dtype(0.),loss_data);
  } else {
    LOG(FATAL) << this->type() 
               << " wrong version " << version;
  }
  caffe_gpu_asum(nthreads, loss_data, &closs);
  wloss+=closs;
  sorter_gpu.Reshape(outter_dim,DIMS,1,1);
  const Dtype* graph = bottom[4]->gpu_data();
  const Dtype* s_val = bottom[1]->gpu_data();
  const int MaxNum = bottom[4]->count()/outter_dim;
  //computing structure preserving constraints for category
  //precomputing minimum or maximun of all Uc
  preComputeGPU<Dtype><<<CAFFE_GET_BLOCKS(outter_dim),CAFFE_CUDA_NUM_THREADS>>>
  (outter_dim, s_val, table_ , graph, sorter_gpu.mutable_gpu_data(), MaxNum);
  Blob<Dtype> s_loss(bottom[1]->shape());
  caffe_gpu_set(s_loss.count(),Dtype(0.),s_loss.mutable_gpu_data());
  Dtype* sloss_data = s_loss.mutable_gpu_data();
  const Dtype* sorter_val = sorter_gpu.gpu_data();
  const int nthread = bottom[1]->count(); // [ batch_size x number of prototypes ]
  //computing structure preserving constraint loss
  spConstraintLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthread),
    CAFFE_CUDA_NUM_THREADS>>>(nthread,s_val,bottom_label,sloss_data,
     graph,table_ ,sorter_val,outter_dim,MaxNum, sp_margin);
  caffe_gpu_asum(nthread, sloss_data, &sloss);
  if( verbose ) {
    
   top[1]->mutable_cpu_data()[0] = wloss/outter_loop;
   top[2]->mutable_cpu_data()[0] = lambda*sloss/bottom[1]->num();
  }
  wloss /= outter_loop;
  wloss+= lambda*(sloss/bottom[1]->num());
  top[0]->mutable_cpu_data()[0] = wloss;
}

template <typename Dtype>
__global__ void spLmeLossPair_Distance_BackwardGPU( const int nthreads, const Dtype* score,
   const Dtype* label, const Dtype* table, Dtype* bottom_diff,
   const int outter_dim, const Dtype margin, const Dtype inner_margin) {
 unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
 if(ix < nthreads) { 
  const int n = ix; //row of score.
  const int label_value = static_cast<int>(label[n]);
  const int index_ = table[label_value*DIM+INDEX]; //category index.
  //check multi labels
  for(int j = 0 ; j < outter_dim ; j++) {
   //if( table[j*DIM+INDEX] == index_ ) continue;
   // pair_wise version of loss 
   if( (ix*outter_dim+label_value) == (ix*outter_dim+j)) {
    continue;
   } else if( table[j*DIM+INDEX] != index_) {
    const Dtype prob = max( Dtype(0),
      margin + score[ix*outter_dim + label_value] - score[ix*outter_dim + j] );
    if( prob > 0){
     bottom_diff[ix*outter_dim + j] -= 1;
     bottom_diff[ix*outter_dim + label_value] += 1;  
    }
   } else if( table[j*DIM+INDEX] == index_ && inner_margin > 0 ) {
    const Dtype prob = max( Dtype(0),
      inner_margin + score[ix*outter_dim + label_value] - score[ix*outter_dim + j] );
    if( prob > 0){
     bottom_diff[ix*outter_dim + j] -= 1;
     bottom_diff[ix*outter_dim + label_value] += 1;  
    }
   }
  }
 }
}


template <typename Dtype>
__global__ void spLmeLossUnary_Distance_BackwardGPU( const int nthreads, const Dtype* score,
   const Dtype* label, const Dtype* table, Dtype* bottom_diff,
   const int outter_dim, const Dtype margin, const Dtype inner_margin) {
 unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
 if(ix < nthreads) { 
  const int n = ix; //row of score.
  const int label_value = static_cast<int>(label[n]);
  const int index_ = table[label_value*DIM+INDEX]; //category index.
  //check multi labels
  for(int j = 0 ; j < outter_dim ; j++) {
   //if( table[j*DIM+INDEX] == index_ ) continue;
   // pair_wise version of loss 
   if( (ix*outter_dim+label_value) == (ix*outter_dim+j)) {
    const Dtype prob = max( Dtype(0),
      score[ix*outter_dim + label_value] - margin);
    if( prob > 0){
     bottom_diff[ix*outter_dim + j] += 1;
    } 
   } else if( table[j*DIM+INDEX] != index_) {
    const Dtype prob = max( Dtype(0),
      inner_margin - score[ix*outter_dim + j] );
    if( prob > 0){
     bottom_diff[ix*outter_dim + j] -= 1;
    } 
   } else if( table[j*DIM+INDEX] == index_ && inner_margin > 0 ) {
    const Dtype prob = max( Dtype(0),
      inner_margin - score[ix*outter_dim + j] );
    if( prob > 0){
     bottom_diff[ix*outter_dim + j] -= 1;
    } 
   }
  }
 }
}


template <typename Dtype>
void spLmeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2] || propagate_down[3] || propagate_down[4]) {
   LOG(FATAL) << this->type() 
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
   const int num = bottom[0]->num();
   const Dtype scale = top[0]->cpu_diff()[0]/num;
   const Dtype* score = bottom[0]->gpu_data();
   const Dtype* bottom_label = bottom[2]->gpu_data();
   const Dtype* table_  = bottom[3]->gpu_data(); // information table
   const int outter_loop = bottom[0]->num(); //batch_size;
   const int outter_dim  = bottom[0]->count()/outter_loop; // total number of prototypes;

   const int nthreads = bottom[0]->num(); 
   //initialize
   Dtype* bottom_diff = bottom[0]->mutable_gpu_diff(); 
   caffe_gpu_set(bottom[0]->count(),Dtype(0.),bottom[0]->mutable_gpu_diff());
   //select the version of loss
   if( version == "pair" ) {
    spLmeLossPair_Distance_BackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, score , bottom_label, table_, 
         bottom_diff, outter_dim, margin,inner_margin);  
    caffe_gpu_scal(bottom[0]->count(), scale , bottom[0]->mutable_gpu_diff());
   } else if( version == "unary" ) {
    spLmeLossUnary_Distance_BackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, score , bottom_label, table_, 
         bottom_diff, outter_dim, margin,inner_margin);  
    caffe_gpu_scal(bottom[0]->count(), scale , bottom[0]->mutable_gpu_diff());

   } else if( version == "verify") {
   
   } else {
    LOG(FATAL) << this->type() 
               << " wrong version " << version;
   }
  }
  //back propagate with respect to structure preserving constraint.
  if (propagate_down[1]) {
   const int num = bottom[1]->num();
   const Dtype scale = lambda*top[0]->cpu_diff()[0]/num;
   const Dtype* graph  = bottom[4]->cpu_data();
   const Dtype* s_value  = bottom[1]->cpu_data();
   const Dtype* bottom_label = bottom[2]->cpu_data();
   const Dtype* table_  = bottom[3]->cpu_data(); 
   //const int outter_loop = bottom[0]->num(); //batch_size;
   const int outter_loop = bottom[0]->count()/bottom[0]->num(); //batch_size;
   //const int outter_dim  = bottom[0]->count()/outter_loop; // total number of prototypes
   const int outter_dim  = outter_loop; // total number of prototypes
   const int MaxNum = bottom[4]->count()/outter_dim;  // Max number of prototypes
   const Dtype* sorter_val = sorter_gpu.cpu_data();
   Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
   caffe_set(bottom[1]->count(), Dtype(0), bottom_diff);

   for( int i = 0 ; i < outter_loop ; i++) {
    //const int label = static_cast<int>(bottom_label[i]);
    const int label = i;
    if( label == -1 ) continue;
    const int label_index = table_[label*DIM+INDEX]; //category index
    for(int j = 0; j < outter_dim ; ++j) { 
     if( table_[j*DIM+INDEX] == label_index && graph[label*MaxNum + j - int(table_[j*DIM+OFF])] ) 
        continue; //except nearest neighbors.
     const Dtype prob = std::max( Dtype(0),
       sp_margin + sorter_val[label*DIMS] - s_value[label*outter_dim + j] );
     if( prob > 0 ) {
      bottom_diff[ label*outter_dim + int(sorter_val[label*DIMS+1])] += 1;
      bottom_diff[ label*outter_dim + j] -= 1;
     }
    }
   }
   caffe_scal(bottom[1]->count(), scale, bottom_diff);
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(spLmeLossLayer);

}  // namespace caffe
