#include <iostream>
#include <stdio.h>
#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

#define FILE "matrix_AA.csv"

int main(){

   	ifstream csvFile;
   	csvFile.open(FILE);
	vector<vector<double>> T_vector = {};
	if (csvFile){
		string line;
		// Read number of vertices
		getline(csvFile, line);
		int vertices_number = stoi(line);
		double *T_static[vertices_number];
		double *static_temp;
		int i = 0;
		// Matrix *myMatrix = new Matrix(vertices_number, vertices_number);
		// Load vertices and connections
		while (getline(csvFile, line)){
			stringstream ss(line);
			string element;
			vector<double> temp = {};
			getline(ss, element, ',');
			// Add probability and connections
        	while (getline(ss, element, ',')) {
            	temp.push_back(stod(element));
        	}
			// Add line to "matrix"
			T_static[i] = (double*) malloc(temp.size()*sizeof(double));
			for (int j = 0; j < temp.size(); j++){
				T_static[i][j] = temp[j];
			}
			i++;
		}
	}
	return 0;
}