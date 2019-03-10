//
// Created by Edoardo Ramalli on 2019-03-10.
//

#ifndef PAGERANKGPU_MATRIX_H
#define PAGERANKGPU_MATRIX_H

#include <vector>

using namespace std;


class Matrix {

private:
    double *mdata;
    int rows;
    int cols;

public:
    Matrix(int rows, int cols);
    void setElement(double value, int i, int j);
    double getElement(int i, int j);
    double* getrawdata();

};


#endif //PAGERANKGPU_MATRIX_H
