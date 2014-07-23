#include <iostream>
#include <stdio.h>
#include <vector>

#define MAX_THREADS 256
#define SIZE 131072

#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define __STOP__(_V) cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop); _V.push_back(time); cudaEventDestroy(start); cudaEventDestroy(stop);
#define __NEXT__(_V) __STOP__(_V) __START__

__global__ void sumall_kernel_shared(float *d_vector, float *d_result)
{
    __shared__ float cache[MAX_THREADS];

    int cacheIdx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[cacheIdx] = i >= SIZE ? 0. : d_vector[i];
    __syncthreads();

    if (i >= SIZE)
        return;

    int padding = blockDim.x/2;
    while (padding != 0)
    {
        if (cacheIdx < padding && i+padding < SIZE)
            cache[cacheIdx] += cache[cacheIdx + padding];

        __syncthreads();
        padding /= 2;
    }

    if (cacheIdx == 0)
        atomicAdd(&d_result[0], cache[0]);
}

__global__ void sumall_kernel_global(float *d_vector, float *d_result)
{ // /!\ it changes d_vector
    int cacheIdx = threadIdx.x;
    int deltaIdx = blockIdx.x * blockDim.x;
    int i = deltaIdx + cacheIdx;

    if (i >= SIZE)
        return;

    int padding = blockDim.x/2;
    while (padding != 0)
    {
        if (cacheIdx < padding && i+padding < SIZE)
            d_vector[i] += d_vector[i + padding];

        __syncthreads();
        padding /= 2;
    }

    if (cacheIdx == 0)
        atomicAdd(&d_result[0], d_vector[deltaIdx]);
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
    std::cout << "SIZE (vs Shared): " << SIZE << std::endl;
    cudaEvent_t start, stop;
    std::vector<float> sharedRun, globalRun;
    float time(0);
    
    cudaFree(0); // Force runtime API context establishment
    float h_vector[SIZE]; // For input and output
    for (unsigned int i(0) ; i!=SIZE ; i++)
        h_vector[i] = i;
    float h_result;
    
    for (unsigned int i(0) ; i!=1000 ; i++)
    {
        float *d_vector, *d_result;
        
        /* SHARED */
        cudaMalloc(&d_vector, SIZE*sizeof(float));
        cudaMalloc(&d_result, sizeof(float));
        cudaMemcpy(d_vector, h_vector, SIZE*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(float));
        __START__
        sumall_kernel_shared<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector, d_result);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP__(sharedRun);
        cudaMemcpy(h_vector, d_vector, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vector);
        cudaFree(d_result);
        
        /* GLOBAL */
        cudaMalloc(&d_vector, SIZE*sizeof(float));
        cudaMalloc(&d_result, sizeof(float));
        cudaMemcpy(d_vector, h_vector, SIZE*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(float));
        __START__
        sumall_kernel_global<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector, d_result);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP__(globalRun);
        cudaMemcpy(h_vector, d_vector, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vector);
        cudaFree(d_result);
    }
    showMean(sharedRun);
    showMean(globalRun);
}    

