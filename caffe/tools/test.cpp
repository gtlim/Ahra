
// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using namespace cv;

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif


  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB("lmdb"));
  //scoped_ptr<db::DB> db2(db::GetDB("lmdb"));
  db->Open(argv[1], db::WRITE);
  //db2->Open(argv[1], db::WRITE);
  shared_ptr<db::Cursor> cursor(db->NewCursor());  
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
  Datum datum;
  db::Cursor* c = cursor.get();
  LOG(INFO) << c->valid();
  datum.ParseFromString(c->value());
  string key_str = c->key();
  LOG(INFO) << key_str;
  LOG(INFO) << datum.label();
  datum.set_label(-1);
  LOG(INFO) << datum.label();
  string out;
  CHECK(datum.SerializeToString(&out));
  char* imageName = "sss";
  namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );
  /*const string& data = datum.data();
  LOG(INFO) << data.size();
  std::vector<uchar> vec_data(data.c_str(), data.c_str() + data.size());
  Mat cv_img_origin = cv::imdecode(vec_data,CV_LOAD_IMAGE_COLOR);
  //Mat cv_img;
  //int width = 256;
  //int height = 256;
  //cv::resize(cv_img_origin, cv_img, cv::Size(width, height));*/
  Mat cv_img_origin = DecodeDatumToCVMat(datum, 1);
  imshow( imageName, cv_img_origin );
  waitKey(0);
  txn->Update(key_str, out);
  txn->Commit();
 
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
