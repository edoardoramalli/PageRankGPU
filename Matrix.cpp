//
// Created by Edoardo Ramalli on 2019-03-10.
//

#include "Matrix.h"
#include <stdexcept>
using  namespace std;

// We represent the matrix as an array of sequential rows, therefore element
// i, j is element i*columns number + j in the array.

Matrix::Matrix(int rows, int cols) {
    this->rows=rows;
    this->cols=cols;
    this->mdata = new double[rows*cols];  // Initializes every element to 0
}

void Matrix::setElement(double value, int i, int j) {
    if (i < rows && j < cols){
        mdata[i * cols + j] = value;
    }
    else throw invalid_argument( "Index out of range" );
}

double Matrix::getElement(int i, int j) {
    if (i < rows && j < cols){
        return mdata[i * cols + j];
    }
    else throw invalid_argument( "Index out of range" );

}


