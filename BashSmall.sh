#!/bin/bash
while getopts l option
do
    case "${option}" in
        l) LOCAL=1;;
    esac
done
DATA=$(date)
if [ "$LOCAL" -eq "1" ]; then
    printf "Local Execution - $DATA\n\n"
    sh BashBase.sh -e ./pagerank_contest_edgelists/graph_small_e.edgelist \
    -v ./pagerank_contest_edgelists/graph_small_v.edgelist\
    -n small.csv\
    -d 0.85\
    -t 0.000001\
    -a ./pagerank_truth_values/small_directed_truth_string\
    -b ./pagerank_truth_values/small_undirected_truth_string
else
    printf "Cluster Execution - $DATA\n\n"
    GPU="srun -N1 --gres=gpu:1  "
    CPU="srun -w slurm-cuda-master "
    sh BashBase.sh -e ./pagerank_contest_edgelists/graph_small_e.edgelist \
    -v ./pagerank_contest_edgelists/graph_small_v.edgelist\
    -n small.csv\
    -d 0.85\
    -t 0.000001\
    -a ./pagerank_truth_values/small_directed_truth_string\
    -b ./pagerank_truth_values/small_undirected_truth_string\
    -c "$CPU"\
    -g "$GPU"
fi
