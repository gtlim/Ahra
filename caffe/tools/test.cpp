
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
  shared_ptr<db::DB> db = shared_ptr<db::DB>((db::GetDB("lmdb")));
  //scoped_ptr<db::DB> db2(db::GetDB("lmdb"));
  db->Open(argv[1], db::WRITE);
  //db2->Open(argv[1], db::WRITE);
  shared_ptr<db::Cursor> cursor(db->NewCursor());  
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
  Datum datum;
   
  db::Cursor* c = cursor.get();
  c->SeekToLast();
  //c->Next();
  //c->SeekToLast();
  LOG(INFO) << c->valid();
  datum.ParseFromString(c->value());
  string key_str = c->key();
  LOG(INFO) <<"Key:" << key_str;
  LOG(INFO) <<"Label:" << datum.label();
  datum.set_label(-1);
  LOG(INFO) << datum.label();
  string out;
  CHECK(datum.SerializeToString(&out));
  const string& data = datum.data();
  //LOG(INFO) << data.size();
  // encode img when put in lmdb.
  //Mat cv_img_origin = DecodeDatumToCVMat(datum, 1);
  //LOG(INFO) << cv_img_origin.channels() << " " << cv_img_origin.rows << " " << cv_img_origin.cols;
  //imshow( imageName, cv_img_origin );
  //waitKey(0);
  
  //check data is change or not.
  for(int i = 0 ; i < 10; i++) LOG(INFO) << int(data[i]);
  //txn->Update(key_str, out);
  //txn->Commit();
  string str = "22";
  string val = txn->Get(str);
  datum.ParseFromString(val);
  LOG(INFO) <<"Retrieved:" << datum.label();
 
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
