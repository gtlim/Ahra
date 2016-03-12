/*
 *
 * @brief this is for generic verison of 
 *  computing distance between instance and proto types
 *  in case of CEDL, there is two types of proto
 *  one is for category proto 
 *   and the other is for attribute 
 *  good luck implemented by gtlim 2015.9.24
 *
 *  please synchronize cpp and cu version at the same time.
 *  this layer is highly optimizied by using cuda implementation
 * 
 */

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/distance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void prepBackwardGPU(const int nthreads,
const Dtype* top_diff, Dtype*scales,
const int row, const int col,const bool mode ) {
 CUDA_KERNEL_LOOP(index,nthreads){ 
  for( int iy = 0 ; iy < col ; iy++){
     Dtype scale = ( mode ) ? top_diff[index*col+iy] : top_diff[iy*row+index];
     scales[index] += scale;
  }
 }
}

template <typename Dtype>
__global__ void DistanceLayerBackwardGPU(const int nthreads,
const Dtype* x_data, const Dtype* prepdata, const Dtype* top_diff, 
const Dtype*scales, Dtype* bottom_diff, const int dim) {
 CUDA_KERNEL_LOOP(index,nthreads){
  const int ix = index/dim; // axis of M
  bottom_diff[index] = Dtype(2.)*(scales[ix]*x_data[index] - prepdata[index]);
 }
}

// computing summation.
template <typename Dtype>
__global__ void prepSummationGPU(const int nthreads,
const Dtype* in, Dtype* out, const int dim) {
 CUDA_KERNEL_LOOP(index,nthreads){ 
  out[index] = 0;
  for( int iy = 0 ; iy < dim ; iy++){
     out[index]+= in[index*dim+iy]; 
  }
 }
}

// computing average.
template <typename Dtype>
__global__ void bsxfunPlusGPU(const int nthreads,
const Dtype* x,const Dtype* y, Dtype* out, const int dim,const int mode) {
 CUDA_KERNEL_LOOP(index,nthreads){ 
  int pos;
  if( mode == 0 ) {
   // row in this case
   pos = index/dim;
  } else if( mode == 1) {
   // column in this case
   pos = index%dim;
  }
  out[index] = x[pos] + y[index]; 
 }
}


template <typename Dtype>
void DistanceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {

  const Dtype* img_feat = bottom[0]->gpu_data(); 
  const Dtype* proto = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int dim = bottom[0]->count()/bottom[0]->num(); //assume that bottom[0] & bottom[1] dimension is same.
  CHECK_EQ(bottom[0]->count()/bottom[0]->num(),bottom[1]->count()/bottom[1]->num()) <<
   "W Dimension and U Dimension is not computable";
  //computing Distance between proto and img_feat

  Blob<Dtype> Y(bottom[0]->shape());
  Blob<Dtype> Y2(bottom[0]->num(),1,1,1);
  Blob<Dtype> X(bottom[1]->shape());
  Blob<Dtype> X2(bottom[1]->num(),1,1,1);
  Blob<Dtype> Z(M_,N_,1,1);
  Blob<Dtype> M(M_,N_,1,1);
  Dtype* y = Y.mutable_gpu_data();  
  Dtype* x = X.mutable_gpu_data();  
  Dtype* z = Z.mutable_gpu_data();  

  caffe_gpu_mul(bottom[0]->count(),img_feat,img_feat,y);
  prepSummationGPU<Dtype><<< CAFFE_GET_BLOCKS(bottom[0]->num()),
      CAFFE_CUDA_NUM_THREADS >>>(bottom[0]->num(),Y.gpu_data(),Y2.mutable_gpu_data(),dim);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          img_feat, proto, (Dtype)0., z);   
  caffe_gpu_scal(Z.count(),Dtype(-2),Z.mutable_gpu_data());
  bsxfunPlusGPU<Dtype><<< CAFFE_GET_BLOCKS(M.count()),
      CAFFE_CUDA_NUM_THREADS >>>(M.count(),Y2.gpu_data(),Z.gpu_data(),M.mutable_gpu_data(),bottom[1]->num(),0);
  caffe_gpu_mul(bottom[1]->count(),proto,proto,x);
  prepSummationGPU<Dtype><<< CAFFE_GET_BLOCKS(bottom[1]->num()),
      CAFFE_CUDA_NUM_THREADS >>>(bottom[1]->num(),X.gpu_data(),X2.mutable_gpu_data(),dim);
  bsxfunPlusGPU<Dtype><<< CAFFE_GET_BLOCKS(M.count()),
      CAFFE_CUDA_NUM_THREADS >>>(M.count(),X2.gpu_data(),M.gpu_data(),top_data,bottom[1]->num(),1);
  
}

template <typename Dtype>
void DistanceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* img_feat = bottom[0]->gpu_data();
  const Dtype* proto = bottom[1]->gpu_data();
  if(propagate_down[0] ) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Blob<Dtype> prep(M_,K_,1,1),scales(M_,1,1,1);
    Blob<Dtype> X(M_,1,1,1);
    Dtype* pdata = prep.mutable_gpu_data();
    Dtype* sdata = scales.mutable_gpu_data();
    caffe_gpu_set(prep.count(),Dtype(0.),pdata);
    caffe_gpu_set(scales.count(),Dtype(0.),sdata);
    caffe_gpu_set(bottom[0]->count(),Dtype(0.),bottom_diff);
    // backprop to img_feat
    // compute corresponding values
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                          top_diff, proto, (Dtype)0., pdata);   
    // pre-compute scales
    prepBackwardGPU<Dtype><<< CAFFE_GET_BLOCKS(M_),
      CAFFE_CUDA_NUM_THREADS >>>(M_,top_diff,sdata,M_, N_,1);
    DistanceLayerBackwardGPU<Dtype><<< CAFFE_GET_BLOCKS(M_*K_),
      CAFFE_CUDA_NUM_THREADS >>>(M_*K_,img_feat, pdata,top_diff,sdata,bottom_diff,K_);
  }
  if(propagate_down[1] ) {   
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
    Blob<Dtype> prep(N_,K_,1,1),scales(N_,1,1,1);
    Blob<Dtype> X(N_,1,1,1);
    Dtype* pdata = prep.mutable_gpu_data();
    Dtype* sdata = scales.mutable_gpu_data();
    caffe_gpu_set(prep.count(),Dtype(0.),pdata);
    caffe_gpu_set(scales.count(),Dtype(0.),sdata);
    caffe_gpu_set(bottom[1]->count(),Dtype(0.),bottom_diff);
    //backprop to proto
    // compute corresponding values
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
                          top_diff, img_feat, (Dtype)0., pdata);   
    // pre-compute scales
    prepBackwardGPU<Dtype><<< CAFFE_GET_BLOCKS(N_),
      CAFFE_CUDA_NUM_THREADS >>>(N_,top_diff,sdata,N_,M_,0);
    DistanceLayerBackwardGPU<Dtype><<< CAFFE_GET_BLOCKS(N_*K_),
      CAFFE_CUDA_NUM_THREADS >>>(N_*K_,proto,pdata,top_diff,sdata,bottom_diff,K_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DistanceLayer);
}  // namespace caffe
