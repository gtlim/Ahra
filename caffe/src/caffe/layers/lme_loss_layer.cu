/**
 * this is main loss 
 * for category explansion 
 * pair-wise
 **/

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lme_loss_layer.hpp"

namespace caffe {


//computing category loss pair-wise + unary Distance
template <typename Dtype>
__global__ void LmeLossForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, Dtype margin, Dtype sigma) {
 CUDA_KERNEL_LOOP(index, nthreads) {
  const int n = index / dim;    //row
  const int label_value = static_cast<int>(label[n]);
  const int label_index = n*dim + label_value;
  // unary version of loss 
  if( label_index == index ) {
   loss[index] += max( Dtype(0),
    bottom_data[index] - sigma);
  } else if( label_index != index ) {
   loss[index] =+ max( Dtype(0),
    margin - bottom_data[label_index] + bottom_data[index] );
   loss[index] += max( Dtype(0),
    sigma - bottom_data[index]);
  }
 }
}

template <typename Dtype>
void LmeLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype closs,wloss;
  wloss = Dtype(0.);
  //computing category loss 
  Blob<Dtype> loss_data(bottom[0]->shape());
  Dtype* loss_data_ = loss_data.mutable_gpu_data();
  const Dtype* score = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count()/num;
  const int nthreads = num * dim; 
  LmeLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, score, label, loss_data_,
      num, dim , margin,sigma);  
  caffe_gpu_asum(nthreads, loss_data_, &closs);
  wloss+=closs;
  wloss/=num;
  top[0]->mutable_cpu_data()[0] = wloss;
}


template <typename Dtype>
__global__ void LmeLossBackwardGPU( const int nthreads, const Dtype* bottom_data,const Dtype* label, 
                    Dtype* bottom_diff, const int num, const int dim,const Dtype margin,const Dtype sigma) {
 unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
 if(ix < nthreads) { 
  const int label_value = static_cast<int>(label[ix]); 
  const int label_index = dim*ix + label_value;
  for( int iy = 0 ; iy < dim ; iy++){
   const int index = dim*ix + iy;
   if( label_index != index ) {
    //compute pair_wise term
    Dtype prob = max( Dtype(0),
	margin - bottom_data[label_index] + bottom_data[index] );
    if( prob > 0){
       bottom_diff[index] += 1;
       bottom_diff[label_index] -= 1; 
    }
    //compute unary term
    prob = max( Dtype(0),
      sigma - bottom_data[index]);
    if(prob > 0) 
      bottom_diff[index] -= 1;
   } else if ( label_index == index ) {
     Dtype prob = max( Dtype(0),
         bottom_data[index] - sigma);
    if(prob > 0) 
       bottom_diff[index] += 1;
   }        
  }
 }
}

template <typename Dtype>
void LmeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int num = bottom[0]->num();
  const Dtype scale = top[0]->cpu_diff()[0]/num;
  //const Dtype scale = top[0]->cpu_diff()[0]/50;
  //const Dtype scale = 1.0/num;
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* score = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = bottom[0]->count()/num;
    const int nthreads = num;
    //initialize 
    caffe_gpu_set(bottom[0]->count(),Dtype(0.),bottom[0]->mutable_gpu_diff());
    LmeLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, score, label, bottom[0]->mutable_gpu_diff(),
        num, dim , margin,sigma );
    caffe_gpu_scal(bottom[0]->count(), scale , bottom[0]->mutable_gpu_diff());
 }
}

INSTANTIATE_LAYER_GPU_FUNCS(LmeLossLayer);

}  // namespace caffe
