/** 
 *
 * Updating Uc multi-prototype version
 * implemented by gtlim 2015 9.29
 **/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multi_proto_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/hdf5.hpp"
#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

namespace caffe {

template <typename Dtype>
void MultiProtoLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top) {
   
  //number of categories X protos 
  const int num_protos = this->layer_param_.multi_proto_param().num_protos(); 
  //dimension of embedding space 
  const int dim = this->layer_param_.multi_proto_param().num_output();      
  // Check if we need to set up the weights
  const string& source = this->layer_param_.multi_proto_param().source();

  if (this->blobs_.size() > 0) { 
    LOG(INFO) << "Skipping parameter initialization";
  } else if( source.size() > 0 ) {
   LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
   hdf_filenames_.clear();
   std::ifstream source_file(source.c_str());
   if (source_file.is_open()) {
     std::string line;
     while (source_file >> line) {
       hdf_filenames_.push_back(line);
     }
   } else {
     LOG(FATAL) << "Failed to open source file: " << source;
   }
   source_file.close();
   num_files_ = hdf_filenames_.size();
   current_file_ = 0;
   const char* filename = hdf_filenames_[0].c_str();
   hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
   if (file_id < 0) {
     LOG(FATAL) << "Failed opening HDF5 file: " << filename;
   }

   //int top_size = this->layer_param_.top_size();
   //hdf_blobs_.resize(top_size);
   this->blobs_.resize(1);
   vector<int> cproto_shape(2);
   cproto_shape[0] = num_protos;
   cproto_shape[1] = dim;
   this->blobs_[0].reset(new Blob<Dtype>(cproto_shape));

   const int MIN_DATA_DIM = 1;
   const int MAX_DATA_DIM = INT_MAX;

   hdf5_load_nd_dataset(file_id, this->layer_param_.top(0).c_str(),
         MIN_DATA_DIM, MAX_DATA_DIM, this->blobs_[0].get());
   
   herr_t status = H5Fclose(file_id);
   CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
    
   // const Dtype* data = this->blobs_[0]->cpu_data();  
   // for( int i = 0 ; i < 100; i++ ) LOG(INFO) << data[i];

  } else { 
    //Initialize the proto_types for the embedding space 
    vector<int> cproto_shape(2);
    cproto_shape[0] = num_protos;
    cproto_shape[1] = dim;
 
   // allocate memory to proto matrix 
   // initialization is not fixed
   // if random
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(cproto_shape));
    // fill the weights only when random initializaiton.
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.multi_proto_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // parameter initialization
   }
   this->param_propagate_down_.resize(this->blobs_.size(), true);
   done = true;

}

template <typename Dtype>
void MultiProtoLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //simplify the calculation.
  top[0]->Reshape(this->blobs_[0]->shape());   
}
template <typename Dtype>
void MultiProtoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(top[0]->count(),this->blobs_[0]->cpu_data(),top_data);
}

template <typename Dtype>
void MultiProtoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
     // Gradient with respect to category proto types
     Blob<Dtype> NewBlob(this->blobs_[0]->shape());
     caffe_set(NewBlob.count(), Dtype(0), NewBlob.mutable_cpu_data());
     caffe_add(top[0]->count(),top[0]->cpu_diff(),this->blobs_[0]->cpu_diff(),NewBlob.mutable_cpu_data());
     caffe_copy(top[0]->count(),NewBlob.cpu_data(),this->blobs_[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiProtoLayer);
#endif

INSTANTIATE_CLASS(MultiProtoLayer);
REGISTER_LAYER_CLASS(MultiProto);

}  // namespace caffe
