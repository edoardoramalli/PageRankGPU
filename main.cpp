#include <iostream>
#include <stdio.h>
#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

#define CONNECTIONS "data.csv"
#define PROBABILITIES "prob.csv"

// __global__ ourkernel(){

// }

int main(){

   	ifstream connFile, probFile;
   	connFile.open(CONNECTIONS);
	// probFile.open(PROBABILITIES);
	vector<vector<double>> T_vector = {};
	int *row_ptrs;
	int *col_indices;
	int *connections;
	double *probabilities;
	int nodes_number;
	int rows;
	int col_indices_number;
	int conn_size;

	// if (probFile){
	// 	string line;
	// 	getline(probFile, line);
	// 	nodes_number = stoi(line);
	// 	probabilities = (double*) malloc(nodes_number*sizeof(double));
	// 	for (int i = 0; i < nodes_number; i++){
	// 		getline(probFile, line);
	// 		probabilities[i] = stod(line);
	// 	}
	// 	probFile.close();
	// }
	cout << "Load connections" << endl;;
	if (connFile){
		string line, element;
		
		// Read rows number and allocate vector
		getline(connFile, line);
		rows = stoi(line);
		row_ptrs = (int *) malloc(rows*sizeof(int));
		
		// Store menaningful rows
		getline(connFile, line);
		stringstream ss(line);
		for (int i = 0; i < rows; i++){
			getline(ss, element, ',');
			row_ptrs[i] = stoi(element);
		}

		// Read column indices number and allocate vector
		getline(connFile, line);
		col_indices_number = stoi(line);
		col_indices = (int *) malloc(col_indices_number*sizeof(int));

		// Store column indices
		getline(connFile, line);
		stringstream tt(line);
		for (int i = 0; i < col_indices_number; i++){
			getline(tt, element, ',');
			col_indices[i] = stoi(element);
		}

		// Read data length
		getline(connFile, line);
		conn_size = stoi(line);
		connections = (int *) malloc(conn_size*sizeof(int));

		// Store column indices
		getline(connFile, line);
		stringstream uu(line);
		for (int i = 0; i < conn_size; i++){
			getline(uu, element, ',');
			connections[i] = stoi(element);
		}
		connFile.close();
	}

	double pr[nodes_number];
	for (int i = 0; i < nodes_number; i++){
		pr[i] = 1/(double)nodes_number;
	}
	string hello;
	cin >> hello;

/* 	if (csvFile){
		string line;
		// Read number of vertices
		getline(csvFile, line);
		int vertices_number = stoi(line);
		double *T_static[vertices_number];
		double *static_temp;
		int i = 0;
		int total_connections = 0;
		int max_connections = 0;
		// Matrix *myMatrix = new Matrix(vertices_number, vertices_number);
		// Load vertices and connections
		while (getline(csvFile, line)){
			stringstream ss(line);
			string element;
			vector<double> temp = {};
			getline(ss, element, ',');
			// Retrieve connections
        	while (getline(ss, element, ',')) {
            	temp.push_back(stod(element));
        	}
			// Add line to "matrix"
			T_static[i] = (double*) malloc((temp.size()+1)*sizeof(double));
			T_static[i][0] = temp.size();
			// total_connections += temp.size();
			// if (temp.size() > max_connections){
			// 	max_connections = temp.size();
			// }
			for (int j = 1; j < temp.size(); j++){
				T_static[i][j] = temp[j];
			}
			i++;
		}	
		// cout << "max_connections: " << max_connections <<endl;
		// cout << "total_connections: " << total_connections << endl;
		// cudamemcpy
		// for (int i = 0; i < vertices_number; i++){
		// ourkernel<<1,T_static[i][0]>>();
		// }
		//cout << T_static[3][0] << endl;;
	}
	*/

	return 0;
}