#include <iostream>
#include <stdio.h>

template <unsigned int blockSize> __global__ void test_final(){
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

template <unsigned int blockSize> __global__ void test(){
    printf("Hello from father block %d, thread %d\n", blockIdx.x, threadIdx.x);
    test_final <2> <<<2,2>>> ();
}

int main(){
    test <1> <<<1,2>>>();
    cudaDeviceSynchronize();
    return 0;

}