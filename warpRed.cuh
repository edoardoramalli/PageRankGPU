void deviceReduce(float *in, float* out, int N, int block_threads);

__global__ void deviceReduceWarpAtomicKernel(float *in, float* out, int N);