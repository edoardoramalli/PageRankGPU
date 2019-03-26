#include <iostream>
#include <stdio.h>
using namespace std;

template <unsigned int blockSize> __global__ void piselli(float *data1) {
    printf("%f", *data1);
}


int main(){
    float f = 100.0;
    float *f_gpu;
    int block = 1;
    cudaMalloc(&f_gpu, sizeof(float));
    cudaMemcpy(f_gpu, &f, sizeof(float),cudaMemcpyHostToDevice);
    piselli<1><<<1 , 1>>>(f_gpu);
    cudaDeviceSynchronize();
}