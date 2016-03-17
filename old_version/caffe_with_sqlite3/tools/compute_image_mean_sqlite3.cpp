
#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/db_sqlite3.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_string(tablename, "Images",
    "Table name in sqlite3.");
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a sqlite3 db\n"
        "Usage:\n"
        "    compute_image_mean_sqlite3 [FLAG] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean_sqite3");
    return 1;
  }
  SQLITECursor* DB_;
  string dbname = argv[2];
  string tablename = FLAGS_tablename;
  vector<string> flags;
  flags.push_back("Id");
  flags.push_back("Data");
  flags.push_back("Label");
  // this is for distinguish unseen and seen.
  flags.push_back("Tag");
  DB_ = new SQLITECursor(dbname,tablename,flags);
  /* Check database */
  sqlite3 *db;
  int rc = sqlite3_open(dbname.c_str(), &db);
  if( rc ){
   LOG(FATAL) << "Can't open database: " << sqlite3_errmsg(db);
  } else {
   LOG(INFO) << "Opened database successfully: " << dbname;
  }
  sqlite3_close(db);
 
  const bool is_color = !FLAGS_gray;
 
  int new_height = std::max<int>(0, FLAGS_resize_height);
  int new_width = std::max<int>(0, FLAGS_resize_width);

  int new_channels;
  if( is_color ) {
   new_channels = 3;
  } else { 
   new_channels = 1;
  }
  BlobProto sum_blob;
  int count = 0;
  vector<int> lines_ = DB_->LookingDB();
  int lines_id_ = 0;
  Datum datum;
  cv::Mat cv_img = DB_->SelectImage(lines_[lines_id_],
                                    new_height, new_width, is_color);
  CVMatToDatum(cv_img, &datum);

  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(new_height);
  sum_blob.set_width(new_width);
  const int data_size = new_channels * new_height * new_width;
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  while (lines_id_ < lines_.size()) {
    Datum datum;
    cv::Mat cv_img = DB_->SelectImage(lines_[lines_id_],
                                    new_height, new_width, is_color);
    CVMatToDatum(cv_img, &datum);
    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    lines_id_++;
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  if (argc == 3) {
    LOG(INFO) << "Write to " << argv[2];
    WriteProtoToBinaryFile(sum_blob, argv[2]);
  }
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> mean_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
