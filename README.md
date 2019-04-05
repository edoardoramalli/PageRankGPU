# PageRankGPU

**PageRank** algorithm implementation in **C++** exploiting **CUDA** to access NVIDIA's GPUs parallel computing capabilities.

### Prerequisites

* CUDA-enabled GPU device with compute capabilities 3.5 or higher
* CUDA toolkit 9.x
* Python3
* g++ 6.x


## Bash Files
In the repository are provided three bash file that should simplify the execution of all procedure. Otherwise, if you prefer a custom execution, step by step, you can follow the four steps above.
- **BaseBash.sh** is a base bash file that allows you to automize all the phase, specifiyng the input parameters. `sh BashBase.sh -n testName -v vertexPath -e edgePath -t thresholdValue -d dampingFactor [-a test1Path] [-b test2Path] [-c cpuCommand] [-g gpuCommand]`
	* `-n testName` name of the test. It will be used to create final and intermediate `.csv` files.
	* `-v vertexPath` vertices file path.
    * `-e edgePath` edges file path.
    * `-t thresholdValue` specify custom precision error threshold.
    * `-d dampingFactor` specify custom damping value.
    * `-a test1Path` pageRank file path to be compared with the compute one.
    * `-b test2Path` vertices file path to be compared with the compute one.
    * `-c cpuCommand` optional cpu command.
    * `-g gpuCommand` optional gpu command.
- **BashSmall** is a bash file that simplify the execution. It relies on the fact that in the execution folder of `sh BashSmall.sh [-l]` is present a folder `pagerank_contest_edgelists` within `graph_small_e.edgelist` and `graph_small_v.edgelist` files that represent the `.csv` files for edges and vertices respectively. The bash file also uses two files `small_directed_truth_string` and `small_undirected_truth_string` in the folder `pagerank_truth_values` that are the "truth" value of the pageRank with which they will be compared.
	* `-l` is a flag that, if presents, means that the execution of the procedure will be local an not in a remote cluster. Otherwise, if this is not present, the `cpuCommand` is`srun -w slurm-cuda-master` and the `gpuCommand` is `srun -N1 --gres=gpu:1 `
    
- **BashFull** as for BashSmall.sh but this time it uses the full data set with the correct relate files.

## 1. Preprocessing

0. **Download dataset** with `sh download_edgelists.sh`, the script will create a `"pagerank_contest_edgelists"` subdirectory in the current directory.
1. **Elaborate dataset** with `python3 ElaborateDataset.py [-v vertexPath -e edgePath|-s|-f] [-o]` 
    * `-v vertexPath` vertices file path.
    * `-e edgePath` edges file path.
    * `-s` use default `"graph_small_e.edgelist"` and files `"graph_small_v.edgelist"` in folder extracted at step 1 as input and save in the current directory a `data_small.csv` file with the processed dataset.
    * `-f` use default `"graph_full_e.edgelist"` and files `"graph_full_v.edgelist"` in folder extracted at step 1 as input and save in the current directory a `data_full.csv` file with the processed dataset.
    * `-o outputPath` specify custom target file for dataset output.
 



## 2. Compilation and computation

1. Compile the sources using `nvcc -arch=sm_35 -rdc=true -lcudadevrt main.cu handleDataset.cpp -o pagerank  -use_fast_math -std=c++11`. If your GPU does not support compute capabilities 3.5 or higher, the compilation will fail. This is required in order to exploit relocatable device code.
2. Run the algorithm computation using `./pagerank [-i inputPath |-s|-f] [-o] [-d] [-t]`
	* `-i inputPath` input CSV dataset file.
    * `-s` uses as input `"data_small.csv"` and `"pk_data_small.csv"` as output.
    * `-f` uses as input `"data_full.csv"` and `"pk_data_full.csv"` as output.
    * `[-o outputPath]` specify custom target file for results output.
    * `-d` specify custom damping value. Defaults to *0.85*
    * `-t` specify custom precision error threshold. Defaults to *10e-6*


## 3. Postprocessing

1. **Elaborate Result** with `python3 GenerateResult.py [-v vertexPath -o pageRankPath -p pageRankPath|-s|-f] [-o]`
    * `-v vertexPath` vertices file path.
    * `-s` use default `"pk_data_small.csv"` and files `"graph_small_v.edgelist"` and save in the current directory a `result_data_small.csv` file with the processed pagerank result associated with the vertex name.
    * `-f` use default `"pk_data_full.csv"` and files `"graph_full_v.edgelist"` and save in the current directory a `result_data_full.csv` file with the processed pagerank result associated with the vertex name.
    * `-p pageRankPath` pageRank file path computed at previous phase.
    * `-o outputPath` specify custom target file for dataset output.

## 4. Validation
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
* DataSet **"Small"**: 22 iterations, time to convergence: *0.045* s.
* DataSet **"Full"**: 34 iterations, time to convergence: *6.402* s.

### EVGA GTX 1060 OC (6GB GDDR5), dual-core Intel core i7 4650U and 8 GB of RAM
* DataSet **"Small"**: 22 iterations, time to convergence: *0.034* s.
* DataSet **"Full"**: 34 iterations, time to convergence: *3.666* s.

	


