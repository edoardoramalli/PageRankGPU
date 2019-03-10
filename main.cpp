#include <iostream>
#include <stdio.h>
#include "Matrix.h"

using namespace std;

int main(){

//   FILE *infile = fopen("test.csv", "r");
//    if (infile!= nullptr){
//
//    }

   Matrix *myMat = new Matrix(10, 10);
   for (int i = 0; i < 10 ; i++){
      for(int j = 0; j < 10; j++){
         cout << myMat->getElement(i,j) << " ";
      }
      cout << endl;
   }
   //cout << myMat->getElement(1,9) << endl;

}