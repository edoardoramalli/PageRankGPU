#include <iostream>
#include <stdio.h>
#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

#define FILE "test.csv"

int main(){

   	ifstream csvFile;
   	csvFile.open(FILE);
	if (csvFile){
		string line;
		// Read number of vertices
		getline(csvFile, line);
		int vertices_number = stoi(line);
		int prob;
		// Matrix *myMatrix = new Matrix(vertices_number, vertices_number);
		vector<vector<double>> T_vector = {};
		// Load vertices and connections
		while (getline(csvFile, line)){
			stringstream ss(line);
			string element;
			vector<double> temp = {};
			
			// Add probability and connections
        	while (getline(ss, element, ',')) {
            	temp.push_back(stod(element));
        	}
			// Add line to vector
			T_vector.push_back(temp);
		}
	}

	Matrix *myMat = new Matrix(10, 10);
	for (int i = 0; i < 10 ; i++){
		for(int j = 0; j < 10; j++){
			cout << myMat->getElement(i,j) << " ";
		}
		cout << endl;
	}


   //cout << myMat->getElement(1,9) << endl;

}