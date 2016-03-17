
/* 
 카테고리 익스펜션을 위해 디비를 만드는 툴.
 */

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

#include <sqlite3.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector> 
using namespace std;


int main(int argc, char **argv) {

#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the sqlite3 DataBase\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_sqlite3 ROOTFOLDER/ LISTFILE DB_NAME\n"); 
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_sqlite3");
    return 1;
  }
   int  rc;
   sqlite3 *db;
   char *sq;
   char *zErrMsg = 0;
  /*create table information*/
  /* Open database */
   rc = sqlite3_open(argv[3], &db);
   if( rc ){
      fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
      exit(0);
   }else{
      fprintf(stdout, "Opened database successfully\n");
   }

   /* Create SQL statement */
   sq  = "CREATE TABLE Images("  \
         "ID INT PRIMARY KEY," \
         "Data           BLOB," \
         "Label          INT," \
         "Tag            INT);";

  /* Execute SQL statement */
  rc = sqlite3_exec(db, sq, 0, 0, 0);
  if( rc != SQLITE_OK ){
   LOG(INFO) << "SQL error: " <<  zErrMsg;
   sqlite3_free(zErrMsg);
   return 1;
  }else{
    LOG(INFO) << "Table created successfully";
  }
  sqlite3_close(db);
  char *err_msg = 0;    
  rc = sqlite3_open(argv[3], &db);
  if (rc != SQLITE_OK) {
      LOG(INFO) << "Cannot open database: " << sqlite3_errmsg(db);
      sqlite3_close(db);         
      return 1;
  } else { 
       LOG(INFO) << "Opened database successfully";
  }
  
  vector<string> fname;
  vector<int> label;
  const  string& source = argv[2]; 
  LOG(INFO) << "Open source: " << argv[2];
  std::ifstream infile(source.c_str());
  string filename;
  int label_;
  while (infile >> filename >> label_) {
    fname.push_back(filename);
    label.push_back(label_);
  }    
  LOG(INFO) <<"Total Images: " <<  fname.size();
  for(int i = 0 ; i < fname.size() ;i++) {
    FILE *fp = fopen(fname[i].c_str(), "rb");
    LOG(INFO) << i;
    if (fp == NULL) { 
        LOG(WARNING)<< "Cannot open image file";    
    }    
    fseek(fp, 0, SEEK_END);
    if (ferror(fp)) {
        LOG(WARNING) << "fseek() failed";
    }   
    int flen = ftell(fp);
    if (flen == -1) {
        LOG(WARNING) << "error occurred";
    }
    fseek(fp, 0, SEEK_SET);
    if (ferror(fp)) {
        LOG(WARNING) << "fseek() failed";
    }   
    char data[flen+1];
    int size = fread(data, 1, flen, fp);
    if (ferror(fp)) {
        LOG(WARNING) << "fread() failed";
    }   
    int r = fclose(fp);
    if (r == EOF) {
        LOG(WARNING)<< "Cannot close file handler";
    }   
    /*char *err_msg = 0;    
    int rc = sqlite3_open(argv[3], &db);
    if (rc != SQLITE_OK) {
        LOG(INFO) << "Cannot open database: " << sqlite3_errmsg(db);
        sqlite3_close(db);        
        return 1;
    } else { 
        //LOG(INFO) << "Opened database successfully";
    }*/
   
    
    sqlite3_stmt *pStmt;

    char *sql = "INSERT INTO Images(Data) VALUES(?)";
    
    rc = sqlite3_prepare(db, sql, -1, &pStmt, 0);
    
    if (rc != SQLITE_OK) {
        LOG(WARNING) <<  "Cannot prepare statement: " <<  sqlite3_errmsg(db);
    }    
    
    sqlite3_bind_blob(pStmt, 1, data, size, SQLITE_STATIC);    
    
    rc = sqlite3_step(pStmt);
    
    if (rc != SQLITE_DONE) {
        
        LOG(INFO) << "execution failed: " <<  sqlite3_errmsg(db);
    }
    /* query is always Images */ 
    char query[100];
    snprintf ( query, 100, "update Images set Label = %d where Id = %d",
                label[i],i+1);
    sqlite3_exec(db,query,0, 0, 0);
    snprintf ( query, 100, "update Images set Tag = %d where Id = %d",
                0,i+1);
    sqlite3_exec(db,query,0, 0, 0);
    sqlite3_finalize(pStmt);  

    if ( i % 1000 == 0) LOG(INFO) << "Processed " << i << " files.";
    //sqlite3_close(db);
 }
 sqlite3_close(db);
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
 return 0;
}
