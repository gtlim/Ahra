#ifndef CAFFE_UTIL_DB_SQLITE3_HPP
#define CAFFE_UTIL_DB_SQLITE3_HPP

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include <sqlite3.h>
#include <string>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;

class SQLITECursor {

 public:
  SQLITECursor(const string dbname, const string tablename,const vector<string> flags){
   table = tablename;
   Id = 0;
   Data = 1;
   Label = 2;
   Mode = 3;
   if( flags.size() != 4) LOG(FATAL) << "Wrong input";
   this->flags = flags;
   curr_id_ = 0;
   LOG(INFO) <<"Init "<< dbname << " table: " << tablename;
   db_name = dbname;
  }
  
  // InsertImage to db which is annotated by cnn.
  void InsertImage(const string imgname);  
  // Select Image from db.
  cv::Mat SelectImage(const int,const int,const int,const bool);
  // Select Label from db.
  int SelectLabel(const int);
  // Looking DB and Get all Ids.
  vector<int> LookingDB();
  // UpdateMode in db
  void UpdateMode(const int); 
  // UpdateLabel in db
  void UpdateLabel(const int,const int); 
  
 private:
  string db_name;
  string table;
  vector<string> flags;
  int Data, Label, Id, Mode;
  int curr_id_;
  vector<int> line_;
};

#endif  
