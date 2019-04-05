#!/bin/bash

BLACK='\033[0;30m'
RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;30m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'

BLACK_BG='\033[0;40m'
RED_BG='\033[0;41m'
GREEN_BG='\033[0;42m'
ORANGE_BG='\033[0;43m'
BLUE_BG='\033[0;44m'
PURPLE_BG='\033[0;45m'
CYAN_BG='\033[0;46m'
GRAY_BG='\033[1;40m'
YELLOW_BG='\033[1;43m'
WHITE_BG='\033[1;47m'

BLINKING='\033[5m'
REVERSE='\033[7m'
NC='\033[0m'

while getopts d:t:v:e:n:a:b: option
do
    case "${option}" in
        t) THRESHOLD=${OPTARG};;
        v) VERTEX=${OPTARG};;
        e) EDGE=${OPTARG};;
        d) DAMPING=$OPTARG;;
        n) NAME=$OPTARG;;
        a) TEST1=$OPTARG;;
        b) TEST2=$OPTARG;;
    esac
done

if [ "$THRESHOLD" = "" ]; then
    printf "${PURPLE}Invalid Threshold Value${NC}\n"
    exit 1
fi

if [ "$VERTEX" = "" ]; then
    printf "${PURPLE}Invalid Vertex Path${NC}\n"
    exit 1
fi

if [ "$EDGE" = "" ]; then
    printf "${PURPLE}Invalid Edge Path${NC}\n"
    exit 1

fi

if [ "$DAMPING" = "" ]; then
    printf "${PURPLE}Invalid Damping Value${NC}\n"
    exit 1
fi

if [ "$NAME" = "" ]; then
    printf "${PURPLE}Invalid Name${NC}\n"
    exit 1
fi

if [ ! -f ./$TEST1 ]; then
    printf "${PURPLE}Invalid Test1 Path${NC}\n"
    exit 1
fi

if [ ! -f ./$TEST2 ]; then
    printf "${PURPLE}Invalid Test1 Path${NC}\n"
    exit 1
fi

# if [(( $(echo "$THRESHOLD < 0" |bc -l) ))] || (( $(echo "$THRESHOLD > 1" |bc -l) )) || (( $(echo "$THRESHOLD = 1" |bc -l) )); then
#     printf "${PURPLE}Invalid Threshold Value${NC}\n"
#     exit 1
# fi

# if (( $(echo "$DAMPING < 0" |bc -l) )) || (( $(echo "$DAMPING > 1" |bc -l) )) || (( $(echo "$DAMPING = 1" |bc -l) )); then
#     printf "${PURPLE}Invalid Damping Value${NC}\n"
#     exit 1
# fi

date
printf "\n"

printf "${YELLOW}Execution PageRank${NC}\n\n"

printf "${CYAN}Name Test : $NAME${NC}\n"
printf "${CYAN}Path Vertex : $VERTEX${NC}\n"
printf "${CYAN}Path Edge : $EDGE${NC}\n"
printf "${CYAN}Threshold : $THRESHOLD${NC}\n"
printf "${CYAN}Damping Factor : $DAMPING${NC}\n"
printf "\n"

ELABORATEBASH="data_$NAME"

if [ -f ./$ELABORATEBASH ]; then
    read -p "File $ELABORATEBASH Already Present, Reuse it? (Y) (N)  "  scelta
    if [ "$scelta" = "Y" ]; then
        printf "\n"
        printf "${RED}Reuse Elaborate DataSet (1/6)${NC}\n\n"
    elif [ "$scelta" = "N" ]; then
        printf "\n"
        printf "${RED}Elaborate DataSet (1/6)${NC}\n\n"
        python3 ElaborateDataSet.py -v ${VERTEX} -e ${EDGE} -d ${DAMPING} -o $ELABORATEBASH
        printf "\n"
    else
        printf "${PURPLE}Invalid Input${NC}\n"
        exit 2
    fi
else
    printf "${RED}Elaborate DataSet (1/6)${NC}\n\n"
    python3 ElaborateDataSet.py -v ${VERTEX} -e ${EDGE} -d ${DAMPING} -o $ELABORATEBASH
    printf "\n"
fi


printf "${RED}Compiling CUDA source file (2/6)${NC}\n"
nvcc -arch=sm_35 -rdc=true -lcudadevrt main.cu handleDataset.cpp -o pagerank -use_fast_math -std=c++11
printf "\n"

PKCOMPUTED="pk_$ELABORATEBASH"

printf "${RED}Execute pagRank Algorithm (3/6)${NC}\n\n"
#srun -N1 --gres=gpu:1  ./pagerank -i $ELABORATEBASH -o $PKCOMPUTED -d $DAMPING -t $THRESHOLD
./pagerank -i $ELABORATEBASH -o $PKCOMPUTED -d $DAMPING -t $THRESHOLD
printf "\n"

PKRESULT="result_$NAME"

printf "${RED}Elaborate Result (4/6)${NC}\n\n"
python3 GenerateResult.py -v $VERTEX -o $PKRESULT -p $PKCOMPUTED
printf "\n"

printf "${RED}Compiling Checker source file (5/6)${NC}\n"
c++ checker.cpp -o checker -std=c++11
printf "\n"

printf "${RED}Execute Verification Algorithm (6/6)${NC}\n\n"

if [ "$TEST1" != "" ]; then
    printf "${ORANGE}TEST 1${NC}\n\n"
    ./checker -c $PKRESULT -t $TEST1 -s
    printf "\n"
fi

if [ "$TEST2" != "" ]; then
    printf "${ORANGE}TEST 2${NC}\n\n"
    ./checker -c $PKRESULT -t $TEST2 -s
    printf "\n"
fi

printf "\n"

printf "${BLINKING}DONE${NC}\n"