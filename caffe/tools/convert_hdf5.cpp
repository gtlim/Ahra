
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
//using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(weights,"",
    "Optional; the pretrained weights in binaryproto type.");
DEFINE_string(name,"Uc",
    "Optional; file name in HDF5 file.");
DEFINE_string(layer,"protos",
    "Optional; target layer to extract learned parameter.");


int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert weights saved as binaryProto to hdf5 file\n"
        "format used as input for matlab.\n"
        "Usage:\n"
        "    convert_hdf5 [FLAGS] NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_hdf5");
    return 1;
  }
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  int num = 0;
  std::string target_layer = FLAGS_layer;
  caffe::NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(FLAGS_weights, &param);
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const caffe::LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name(); 
    // LOG(INFO) << "Copying source layer " << source_layer_name;
    if( source_layer_name == target_layer) {
      LOG(INFO) << "Copying source layer " << source_layer_name;
      const bool kReshape = true;
      std::string FILE(argv[2]);
      Blob<float> source_blob;
      source_blob.FromProto(source_layer.blobs(num), kReshape);
      LOG(INFO) << source_blob.shape_string() << "; target param shape is ";
      FILE += ".h5";
      hid_t file_id;
      const std::string name = FLAGS_name;
      file_id = H5Fcreate(FILE.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      caffe::hdf5_save_nd_dataset<float>( file_id, name , source_blob ,false);
      LOG(INFO) <<"Saving Param to " << argv[2];
      break;
    }
   }
  //std::string filename(argv[2]); 
  //filename+= ".h5";
  //LOG(INFO) << "convert to HDF5 file " << filename;
  //caffe_net.ToHDF5(filename, false); //ignore diff_ datafile.
  LOG(INFO) << "Done.";
  return 0;
}
