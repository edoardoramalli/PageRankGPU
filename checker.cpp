using namespace std;
#include <unistd.h>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>
#include <unordered_map>

double computeDistance(unordered_map<int, double> &truth, unordered_map<int, double> &check) {
  double norm = 0;

  for (auto it : truth) {
    norm += (it.second - check[it.first]) * (it.second - check[it.first]);
  }
  norm = sqrt(norm);
  return norm;
}

double computeDistanceStringIndex(unordered_map<string, double> &truth, unordered_map<string, double> &check) {
  double norm = 0;

  for (auto it : truth) {
    norm += (it.second - check[it.first]) * (it.second - check[it.first]);
  }
  norm = sqrt(norm);
  return norm;
}


unordered_map<int, double> readPagerankIntIndexed(string filename) {
  unordered_map<int, double> result;
  ifstream inputFile(filename.c_str());
  if (inputFile.good()) {
      // Push items into a vector
      string index;
      string value;
      while (inputFile >> index >> value){
        result[atoi(index.c_str())] = atof(value.c_str());
      }
      // Close the file.
      inputFile.close();
  } else {
      cout << "Error!";
  }
  return result;
}

unordered_map<string, double> readPagerankStringIndexed(string filename) {
  unordered_map<string, double> result;
  ifstream inputFile(filename.c_str());
  if (inputFile.good()) {
      // Push items into a vector
      string index;
      string value;
      while (inputFile >> index >> value){
        result[index] = atof(value.c_str());
      }
      // Close the file.
      inputFile.close();
  } else {
      cout << "Error!";
  }
  return result;
}


int main (int argc, char *argv[]) {

  string truth_file = "";
  string checked_file = "";
  string dictionary_file = "";

  bool string_index = false;

  int opt;
  while ((opt = getopt(argc,argv,"t:c:s")) != EOF){
    switch(opt)
    {
      case 't':
        truth_file = optarg;
        break;
      case 'c':
        checked_file = optarg;
        break;
      case 's':
        string_index = true;
        break;
      default: return 0;
    }
  }

  if(truth_file == "" || checked_file == ""){
    cout << "missing input files" << endl;
    return 0;
  }

  double norm = 0;
  double pct_error = 0;
  if(string_index) {
    unordered_map<string,double> truth = readPagerankStringIndexed(truth_file);
    unordered_map<string,double> check = readPagerankStringIndexed(checked_file);

    if(truth.size() != check.size()){
      cout << "different size!" << endl;
    }
    //compute distance between the two things
    norm = computeDistanceStringIndex(truth, check);
  } else {
    unordered_map<int,double> truth = readPagerankIntIndexed(truth_file);
    unordered_map<int,double> check = readPagerankIntIndexed(checked_file);

    if(truth.size() != check.size()){
      cout << "different size!" << endl;
    }
    //compute distance between the two things
    norm = computeDistance(truth, check);
  }

  cout << "Euclidean distance: " << norm << endl;
  cout << "Accuracy: " << (1 - norm) * 100 << "\%" << endl;
  return 0;

}


