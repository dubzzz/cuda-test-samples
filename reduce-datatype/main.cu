#include <iostream>
#include <stdio.h>
#include <vector>

#define MAX_THREADS 256
#define SIZE 524288

#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define __STOP__(_V) cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop); _V.push_back(time); cudaEventDestroy(start); cudaEventDestroy(stop);
#define __NEXT__(_V) __STOP__(_V) __START__

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
}

template<class T>
__global__ void sumall_kernel(T *d_vector, T *d_result)
{
    __shared__ T cache[MAX_THREADS];

    int cacheIdx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[cacheIdx] = i >= SIZE ? 0. : d_vector[i];
    __syncthreads();

    if (i >= SIZE)
        return;

    int padding = blockDim.x/2;
    while (padding != 0)
    {
        if (cacheIdx < padding)
        cache[cacheIdx] += cache[cacheIdx + padding];

        __syncthreads();
        padding /= 2;
    }

    if (cacheIdx == 0)
        atomicAdd(&d_result[0], cache[0]);
}

void showMean(std::vector<float> v)
{
    float sum(0);
    for (unsigned int i(0) ; i!=v.size() ; i++)
        sum += v[i];
    std::cout << 1000.*sum/v.size() << " microseconds" << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "SIZE (Reduce-Datatype): " << SIZE << std::endl;
    cudaEvent_t start, stop;
    std::vector<float> intRun, floatRun, doubleRun;
    float time(0);
    
    cudaFree(0); // Force runtime API context establishment
    int h_vector_i[SIZE]; // For input and output
    float h_vector_f[SIZE]; // For input and output
    double h_vector_d[SIZE]; // For input and output
    for (unsigned int i(0) ; i!=SIZE ; i++)
    {
        h_vector_i[i] = i;
        h_vector_f[i] = i;
        h_vector_d[i] = i;
    }
    int h_result_i;
    float h_result_f;
    double h_result_d;
    
    for (unsigned int i(0) ; i!=1000 ; i++)
    {
        int *d_vector_i, *d_result_i;
        float *d_vector_f, *d_result_f;
        double *d_vector_d, *d_result_d;
        
        /* INT */
        cudaMalloc(&d_vector_i, SIZE*sizeof(int));
        cudaMalloc(&d_result_i, sizeof(int));
        cudaMemcpy(d_vector_i, h_vector_i, SIZE*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_result_i, 0, sizeof(int));
        __START__
        sumall_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector_i, d_result_i);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP__(intRun);
        cudaMemcpy(h_vector_i, d_vector_i, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_result_i, d_result_i, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_vector_i);
        cudaFree(d_result_i);
        
        /* FLOAT */
        cudaMalloc(&d_vector_f, SIZE*sizeof(float));
        cudaMalloc(&d_result_f, sizeof(float));
        cudaMemcpy(d_vector_f, h_vector_f, SIZE*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_result_f, 0, sizeof(float));
        __START__
        sumall_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector_f, d_result_f);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP__(floatRun);
        cudaMemcpy(h_vector_f, d_vector_f, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_result_f, d_result_f, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vector_f);
        cudaFree(d_result_f);
        
        /* DOUBLE */
        cudaMalloc(&d_vector_d, SIZE*sizeof(double));
        cudaMalloc(&d_result_d, sizeof(double));
        cudaMemcpy(d_vector_d, h_vector_d, SIZE*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_result_d, 0, sizeof(double));
        __START__
        sumall_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector_d, d_result_d);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP__(doubleRun);
        cudaMemcpy(h_vector_d, d_vector_d, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_result_d, d_result_d, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_vector_d);
        cudaFree(d_result_d);
    }
    showMean(intRun);
    showMean(floatRun);
    showMean(doubleRun);
}    

