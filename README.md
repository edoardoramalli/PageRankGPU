# PageRankGPU

**PageRank** algorithm implementation in **C++** exploiting **CUDA** to access NVIDIA's GPUs parallel computing capabilities.

### Prerequisites

* CUDA-enabled GPU device with compute capabilities 3.5 or higher
* CUDA toolkit 9.x
* Python3
* g++ 6.x

INSERIRE DESCRIZIONE DEI VARI PASSAGGI E COSA FANNO. 
DIRE CHE POSSONO ESSERE SKIPPATI USANDO BASH UNICA.
DIRE CHE PER IL CLUSTER C'è UNA BASH SPECIFICA CHE USA IL COMANDO SRUN...

## Preprocessing

0. **Download dataset** with `sh download_edgelists.sh`, the script will create a `"pagerank_contest_edgelists"` subdirectory in the current directory.
1. **Elaborate dataset** with `python3 ElaborateDataset.py [-v vertexPath -e edgePath|-s|-f] [-o]` 
    * `-v vertexPath` vertices file path.
    * `-e edgePath` edges file path.
    * `-s` use default `"graph_small_e.edgelist"` and files `"graph_small_v.edgelist"` in folder extracted at step 1 as input and save in the current directory a `data_small.csv` file with the processed dataset.
    * `-f` use default `"graph_full_e.edgelist"` and files `"graph_full_v.edgelist"` in folder extracted at step 1 as input and save in the current directory a `data_full.csv` file with the processed dataset.
    * `-o outputPath` specify custom target file for dataset output.
 



## Compilation and computation

1. Compile the sources using `nvcc -arch=sm_35 -rdc=true -lcudadevrt main.cu handleDataset.cpp -o pagerank  -use_fast_math -std=c++11`. If your GPU does not support compute capabilities 3.5 or higher, the compilation will fail. This is required in order to exploit relocatable device code.
2. Run the algorithm computation using `./pagerank [-i inputPath |-s|-f] [-o] [-d] [-t]`
	* `-i inputPath` input CSV dataset file.
    * `-s` uses as input `"data_small.csv"` and `"pk_data_small.csv"` as output.
    * `-f` uses as input `"data_full.csv"` and `"pk_data_full.csv"` as output.
    * `[-o outputPath]` specify custom target file for results output.
    * `-d` specify custom damping value. Defaults to *0.85*
    * `-t` specify custom precision error threshold. Defaults to *10e-6*

You can skip these steps by using the provided script to automate this phase `sh compileandrun.sh` and passing the parameters as specified above

## Postprocessing

1. **Elaborate Result** with `python3 GenerateResult.py [-v vertexPath -o pageRankPath -p pageRankPath|-s|-f] [-o]`
    * `-v vertexPath` vertices file path.
    * `-s` use default `"pk_data_small.csv"` and files `"graph_small_v.edgelist"` and save in the current directory a `result_data_small.csv` file with the processed pagerank result associated with the vertex name.
    * `-f` use default `"pk_data_full.csv"` and files `"graph_full_v.edgelist"` and save in the current directory a `result_data_full.csv` file with the processed pagerank result associated with the vertex name.
    * `-p pageRankPath` pageRank file path computed at previous phase.
    * `-o outputPath` specify custom target file for dataset output.

## Validation
1. **Check Result** using the command `c++ checker.cpp -o checker' generate the binary to check the result.
2. Run the code using `./checker -c checkerPath -t truthPath [-s]`
	* `-c checkerPath` path of the file to be checked
	* `-t truthPath` path of the truth file
	* `-s` to indicate that the indices of the two files are strings
# Results
The time is calculated from the first call of the kernel to the completion of the pagerank final values from GPU memory to main memory.
Damping factor selected for the tests is *0.85*.
Precision threshold is *0.000001*.

### CLUSTER FACILITY (GTX 960 with 2GB of GRAM, 8 cores Intel core i7 and 32GB of RAM)
* DataSet **"Small"**: 22 iterations, time to covergence: *0.045* s.
* DataSet **"Full"**: 34 iterations, time to covergence: *6.402* s.

### EVGA GTX 1060 OC (6GB GDDR5), dual-core Intel core i7 4650U and 8 GB of RAM
* DataSet **"Small"**: 22 iterations, time to covergence: *0.034* s.
* DataSet **"Full"**: 34 iterations, time to covergence: *3.666* s.

	

