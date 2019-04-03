#include "handleDataset.h"

void loadDimensions (string datasetPath, int &nodes_number, int &col_indices_number, float &damping, int &empty_len){
    ifstream connFile;
    connFile.open(datasetPath);

    cout << "Load dimensions" << endl;
    if (connFile){
        string line, element;
        
        // Read nodes number 
        getline(connFile, line);
        nodes_number = stoi(line);

        // Read column indices length
        getline(connFile, line);
        col_indices_number = stoi(line);
        
        // Skip row pointers array
        getline(connFile, line);

        // Skip column
        getline(connFile, line);

        // Skip data
        getline(connFile, line);

        // Save "damping" matrix factor
        getline(connFile, line);
        damping = stof(line);
        cout << "READ DAMPING: " << damping << endl;

        // Read empty columns vector length
        getline(connFile, line);
        empty_len = stoi(line);

        //cout << "Empty length: " << empty_len << endl;

        connFile.close();

    }
} 


void loadDataset(string datasetPath, int* row_ptrs, int* col_indices, float* data, int* empty_cols){
    ifstream connFile;
    connFile.open(datasetPath);

    cout << "Load data" << endl;
    if (connFile){
        string line, element;
        
        // Read nodes number and allocate vector
        getline(connFile, line);
        int nodes_number = stoi(line);
        
        // Read row indices number and allocate vector
        getline(connFile, line);
        int col_indices_number = stoi(line);

        // Store row indices
        getline(connFile, line);
        stringstream ss(line);
        for (int i = 0; i < col_indices_number; i++){
            getline(ss, element, ',');
            row_ptrs[i] = stoi(element);
        }

        // Store column indices
        getline(connFile, line);
        stringstream tt(line);
        for (int i = 0; i < col_indices_number; i++){
            getline(tt, element, ',');
            col_indices[i] = stoi(element);

        }

        // Store data
        getline(connFile, line);
        stringstream uu(line);
        for (int i = 0; i < col_indices_number; i++){
            getline(uu, element, ',');
            data[i] = stof(element);
        }

        // Skip "damping" matrix factor
        getline(connFile, line);
        
        // Read empty columns vector length
        getline(connFile, line);
        int empty_len = stoi(line);

        // Store empty columns indices
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


