#include "handleDataset.h"

void loadDimensions (string datasetPath, int &nodes_number, int &col_indices_number, int &conn_size, int &row_len, float &damping, int &empty_len){
    ifstream connFile;
    connFile.open(datasetPath);

    cout << "Load dimensions" << endl;
    if (connFile){
        string line, element;
        
        // Read row_len number 
        getline(connFile, line);
        row_len = stoi(line);
        nodes_number = row_len-1;

        // Skip row pointers array
        getline(connFile, line);


        // Read column indices length
        getline(connFile, line);
        col_indices_number = stoi(line);

        // Skip column indices
        getline(connFile, line);

        getline(connFile, line);
        conn_size = stoi(line);

        // Skip column indices
        getline(connFile, line);

        //cout << "Data size: " << conn_size << "\nColumn indices:  " << col_indices_number << "\nRow pointers:  " << row_len << endl;

        // Save "damping" matrix factor
        getline(connFile, line);
        damping = stof(line);

        // cout << "Damping contribute: " << damping << endl;

        // Read empty columns vector length
        getline(connFile, line);
        empty_len = stoi(line);

        //cout << "Empty length: " << empty_len << endl;

        connFile.close();

    }
} 


void loadDataset(string datasetPath, int* row_ptrs, int* col_indices, float* connections, int* empty_cols){
    ifstream connFile;
    connFile.open(datasetPath);

    cout << "Load connections" << endl;
    if (connFile){
        string line, element;
        
        // Read row_len number and allocate vector
        getline(connFile, line);
        int row_len = stoi(line);
        int nodes_number = row_len-1;
        
        // Store meaningful rows
        getline(connFile, line);
        stringstream ss(line);
        for (int i = 0; i < row_len; i++){
            getline(ss, element, ',');
            row_ptrs[i] = stoi(element);
        }

        // Read column indices number and allocate vector
        getline(connFile, line);
        int col_indices_number = stoi(line);

        // Store column indices
        getline(connFile, line);
        stringstream tt(line);
        for (int i = 0; i < col_indices_number; i++){
            getline(tt, element, ',');
            col_indices[i] = stoi(element);
        }

        // Read data length
        getline(connFile, line);
        int conn_size = stoi(line);

        // Store data
        getline(connFile, line);
        stringstream uu(line);
        for (int i = 0; i < conn_size; i++){
            getline(uu, element, ',');
            connections[i] = stof(element);
        }

        // Save "damping" matrix factor
        getline(connFile, line);
        
        // Read empty columns vector length
        getline(connFile, line);
        int empty_len = stoi(line);

        // Store data
        getline(connFile, line);
        stringstream vv(line);

        for (int i = 0; i < empty_len; i++){
            getline(vv, element, ',');
            empty_cols[i] = stoi(element);
        }

        connFile.close();
    }
}

void storePagerank(float *pk, int size, string exportPath){
    ofstream CSVFile;
    CSVFile.open(exportPath);
    if(CSVFile.is_open()){
        for (int i = 0; i < size; i++){
            CSVFile << pk[i] << endl;
        }
    }
}


