/****
 Err: "library routine called out of sequence"
 you need to call sqlite3_finalize on the statement 
 if you're not keeping a handle to it somewhere. 
 Otherwise you'll get resource leaks and eventually crash.
****/

#include "caffe/util/db_sqlite3.hpp"
#include <stdio.h>

#define INTER 2

void SQLITECursor::InsertImage(const string imgname) {
 FILE *fp = fopen(imgname.c_str(), "rb");
 if (fp == NULL) LOG(FATAL) << "Cannot open image file";    
 fseek(fp, 0, SEEK_END);

 if (ferror(fp)) {        
  int r = fclose(fp);
  if (r == EOF) LOG(FATAL) << "Cannot close file handler";              
  LOG(FATAL)<<"fseek() failed";
 }
  
 int flen = ftell(fp);
    
 if (flen == -1) {       
  fclose(fp);
  LOG(FATAL)<<"Error Occurred";
 }
    
 fseek(fp, 0, SEEK_SET);
    
 if (ferror(fp)) {
  fclose(fp);
  LOG(FATAL)<<"fseek() failed";
 }

 char data[flen+1];
 int size = fread(data, 1, flen, fp);
   
 if (ferror(fp)) {
  fclose(fp);
  LOG(FATAL)<<"fread() failed";
 }
    
 int r = fclose(fp);

 if (r == EOF) {
   LOG(FATAL) << "Cannot close file handler";
 }    
 
 sqlite3 *db;
    
 int rc = sqlite3_open(db_name.c_str(), &db);
    
 if (rc != SQLITE_OK) {
  sqlite3_close(db);
  LOG(FATAL) << "Cannot open database: " << sqlite3_errmsg(db);      
 } else { 
  //LOG(INFO) << "Opened database successfully";
 }
 sqlite3_stmt *pStmt;
 char sql[100];
 snprintf( sql, 100, "INSERT INTO %s(%s) VALUES(?)", table.c_str(),flags[Data].c_str());   
 rc = sqlite3_prepare(db, sql, -1, &pStmt, 0);   
 if (rc != SQLITE_OK) {
   sqlite3_close(db);
   LOG(FATAL) << "Cannot prepare statement: " << sqlite3_errmsg(db);  
 }

 sqlite3_bind_blob(pStmt, 1, data, size, SQLITE_STATIC);           
 rc = sqlite3_step(pStmt);

 sqlite3_finalize(pStmt);    
 sqlite3_close(db);
}

cv::Mat SQLITECursor::SelectImage(const int id,const int height, 
 const int width, const bool is_color) {

 sqlite3 *db;
 int rc = sqlite3_open(db_name.c_str(), &db);
 if (rc != SQLITE_OK) {       
  sqlite3_close(db);
  //return Mat mat;
  LOG(FATAL) << "Cannot open database: " << sqlite3_errmsg(db);      
 }
 char sql[100];
 snprintf( sql, 100, "SELECT %s FROM %s WHERE %s = %d",
     flags[Data].c_str(),table.c_str(),flags[Id].c_str(),id);   
 //LOG(INFO) << sql;
 sqlite3_stmt *pStmt;
 rc = sqlite3_prepare_v2(db, sql, -1, &pStmt, 0);
 if (rc != SQLITE_OK ) {               
  sqlite3_close(db);
  //return Mat mat;
  LOG(FATAL) << "Cannot open database: " << sqlite3_errmsg(db);    
 } 
 rc = sqlite3_step(pStmt);
 int bytes = 0;   
 if (rc == SQLITE_ROW) {
  bytes = sqlite3_column_bytes(pStmt, 0);
 }
 uchar* p = (uchar*)sqlite3_column_blob(pStmt,0); 
 vector<uchar> data(p, p+bytes);
 int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
 cv::Mat cv_img;
 cv::Mat cv_img_origin(cv::imdecode(data, cv_read_flag));
 if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
 } else {
    cv_img = cv_img_origin;
 }

 rc = sqlite3_finalize(pStmt);   
 sqlite3_close(db);

 return cv_img;
}    

int SQLITECursor::SelectLabel(const int id) {
 sqlite3 *db;
 string label;
 if ( sqlite3_open(db_name.c_str(), &db) != SQLITE_OK ) LOG(FATAL) <<"Implmentation wrong"; 
 sqlite3_stmt *statement;
 /* query is always photos_image */ 
 char query[100];
 snprintf ( query, 100, "select %s from %s where %s = %d",
     flags[Label].c_str(),table.c_str(),flags[Id].c_str(),id);
 if ( sqlite3_prepare(db, query, -1, &statement, 0 ) == SQLITE_OK ) {
  int res = 0;
  while ( 1 ){
   res = sqlite3_step(statement);
   if ( res == SQLITE_ROW ) {
     label = (char*)sqlite3_column_text(statement,0);
    //  LOG(INFO) << atoi(label.c_str()) << " ";
   }
   if ( res == SQLITE_DONE || res==SQLITE_ERROR) {
     break;
   }
  }
 } else {
  LOG(FATAL) <<"Could not Open";
 }
 sqlite3_finalize(statement);
 sqlite3_close(db);
 return atoi(label.c_str());
}

void SQLITECursor::UpdateMode(const int id) {
 sqlite3 *db;
 if ( sqlite3_open(db_name.c_str(), &db) != SQLITE_OK ) LOG(FATAL) <<"Implmentation wrong"; 
 /* query is always photos_image */ 
 char query[100];
 snprintf ( query, 100, "update %s set %s = %d where %s = %d",
                table.c_str(),flags[Mode].c_str(),INTER,flags[Id].c_str(),id);
 sqlite3_exec(db,query,0, 0, 0);
 sqlite3_close(db);
}

void SQLITECursor::UpdateLabel(const int id,const int label) {
 sqlite3 *db;
 if ( sqlite3_open(db_name.c_str(), &db) != SQLITE_OK ) LOG(FATAL) <<"Implmentation wrong"; 
 /* query is always photos_image */ 
 char query[100];
 snprintf ( query, 100, "update %s set %s = %d where %s = %d",
                table.c_str(),flags[Label].c_str(),label,flags[Id].c_str(),id);
 sqlite3_exec(db,query,0, 0, 0);
 sqlite3_close(db);
}

vector<int> SQLITECursor::LookingDB() {
 sqlite3 *db;
 vector<int> line_;
 if ( sqlite3_open(db_name.c_str(), &db) != SQLITE_OK ) LOG(FATAL) <<"Implmentation wrong"; 
 sqlite3_stmt *statement;
 /* query is always photos_image */ 
 char query[100];
 snprintf ( query, 100, "select %s from %s where %s >= %d AND %s != %d",
                   flags[Id].c_str(),table.c_str(),flags[Id].c_str(),0,flags[Mode].c_str(),INTER);
 //LOG(INFO) << query;
 if ( sqlite3_prepare(db, query, -1, &statement, 0 ) == SQLITE_OK ) {
  int res = 0;
  while ( 1 ){
   res = sqlite3_step(statement);
   if ( res == SQLITE_ROW ) {
     string id = (char*)sqlite3_column_text(statement,0);
     line_.push_back(atoi(id.c_str()));
   }
   if ( res == SQLITE_DONE || res==SQLITE_ERROR) {
     break;
   }
  }
 } else {
  LOG(FATAL) <<"Could not Open";
 }
 sqlite3_finalize(statement);
 sqlite3_close(db);
 return line_;
}



