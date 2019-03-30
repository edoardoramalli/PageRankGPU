#ifndef _HANDLEDATASET_H_
#define _HANDLEDATASET_H_

#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

void loadDataset(string datasetPath, int *row_ptrs, int *col_indices, float *connections, int *empty_cols);

void loadDimensions (string datasetPath, int &nodes_number, int &col_indices_number, int &conn_size, int &row_len, float &damping, int &empty_len);

void storePagerank(float *pk, int size, string exportPath);

#endif
