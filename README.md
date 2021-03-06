# PageRankGPU

**PageRank** algorithm implementation in **C++** exploiting **CUDA** to access NVIDIA's GPUs parallel computing capabilities.

### Prerequisites

* CUDA-enabled GPU device with compute capabilities 3.5 or higher
* CUDA toolkit 9.x
* Python3 and tqdm library
* g++ 6.x


## Bash Files
In the repository are provided three bash scripts that should simplify the execution of the whole procedure. Otherwise, if you prefer a step by step custom execution, you can follow the four steps below.
- **BaseBash.sh** is a base bash file that allows you to automate all the phases, by specifiyng the input parameters. 
`sh BashBase.sh -n testName -v vertexPath -e edgePath -t thresholdValue -d dampingFactor [-a test1Path] [-b test2Path] [-c cpuCommand] [-g gpuCommand]`
	* `-n testName` name of the test. It will be used to create final and intermediate `.csv` files.
	* `-v vertexPath` vertices file path.
    * `-e edgePath` edges file path.
    * `-t thresholdValue` specify custom precision error threshold.
    * `-d dampingFactor` specify custom damping factor value.
    * `-a test1Path` true PageRank file path to be compared with the computed one.
    * `-b test2Path` vertices file path to be compared with the computed one.
    * `-c cpuCommand` optional cpu command.
    * `-g gpuCommand` optional gpu command.
- **BashSmall** is a bash script that simplifies the execution. It relies on the assumption that in the execution folder of `sh BashSmall.sh [-l]` is present a subfolder `pagerank_contest_edgelists` containing `graph_small_e.edgelist` and `graph_small_v.edgelist` files, that represent the `.csv` files for edges and vertices respectively. 
The bash file also uses two files, `small_directed_truth_string` and `small_undirected_truth_string`, located in  `pagerank_truth_values` folder, which are the "truth" value of the pageRank against which they will be compared.
	* `-l` is a flag that, if present, specifies that the execution of the procedure will be local and not in a remote cluster (which has to support slurm computing facility). Otherwise, if this is not present, the `cpuCommand` default is`srun -w slurm-cuda-master` and the `gpuCommand`'s is `srun -N1 --gres=gpu:1 `
    
- **BashFull** as for BashSmall.sh, but in this case it uses the full dataset with the correlated files.

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
    * `-s` use default dataset name `"pk_data_small.csv"` and files `"graph_small_v.edgelist"` and save in the current directory a `result_data_small.csv` file containing the processed pagerank result associated with the vertex name.
    * `-f` use default dataset name `"pk_data_full.csv"` and files `"graph_full_v.edgelist"` and save in the current directory a `result_data_full.csv` file containing the processed pagerank result associated with the vertex name.
    * `-p pageRankPath` pageRank file path computed at previous phase.
    * `-o outputPath` specify custom target file for dataset output.

## 4. Validation
0. **Download Truth Value of pageRank** with `sh download_truth_values.sh`, the script will create a `"pagerank_truth_values"` subdirectory in the current directory with "truth" pageRank values.
1. **Check Result** using the command `c++ checker.cpp -o checker' generate the binary to check the result.
2. Run the code using `./checker -c checkerPath -t truthPath [-s]`
	* `-c checkerPath` path of the file to be checked.
	* `-t truthPath` path of the truth file.
	* `-s` to indicate that the indices of the two files are strings.
# Results
The computation time is calculated from the first call of the kernel to the completion of the copy of PageRank's final values from GPU memory to main memory.
Damping factor selected for the tests is *0.85*.
Precision threshold is *0.000001*.

### CLUSTER FACILITY (GTX 960 with 2GB of GRAM, 8 cores Intel core i7 and 32GB of RAM)
* DataSet **"Small"**: 22 iterations, time to convergence: *0.045* s.
* DataSet **"Full"**: 34 iterations, time to convergence: *6.402* s.

### EVGA GTX 1060 OC (6GB GDDR5), dual-core Intel core i7 4650U and 8 GB of RAM
* DataSet **"Small"**: 22 iterations, time to convergence: *0.034* s.
* DataSet **"Full"**: 34 iterations, time to convergence: *3.666* s.

	



