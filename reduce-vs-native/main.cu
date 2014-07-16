#include <iostream>
#include <stdio.h>
#include <vector>

#define MAX_THREADS 256
#define SIZE 524288

#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define __STOP__(_V) cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop); _V.push_back(time); cudaEventDestroy(start); cudaEventDestroy(stop);
#define __NEXT__(_V) __STOP__(_V) __START__
#define __START_BIS__ cudaEventCreate(&startBis); cudaEventCreate(&stopBis); cudaEventRecord(startBis, 0);
#define __STOP_BIS__(_V) cudaEventRecord(stopBis, 0); cudaEventSynchronize(stopBis); cudaEventElapsedTime(&time, startBis, stopBis); _V.push_back(time); cudaEventDestroy(startBis); cudaEventDestroy(stopBis);
#define __NEXT_BIS__(_V) __STOP_BIS__(_V) __START_BIS__

__global__ void sumall_kernel(float *d_vector, float *d_result)
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
    std::cout << 1000.*sum/v.size() << " milliseconds" << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "SIZE (vs Reduce-Native): " << SIZE << std::endl;
    cudaEvent_t start, stop, startBis, stopBis;
    std::vector<float> cudaRun, cudaKRun, nativeRun;
    float time(0);
    
    cudaFree(0); // Force runtime API context establishment
    float h_vector[SIZE]; // For input and output
    for (unsigned int i(0) ; i!=SIZE ; i++)
        h_vector[i] = i;
    float h_result;
    
    for (unsigned int i(0) ; i!=1000 ; i++)
    {
        float *d_vector, *d_result;
        
        __START__
        cudaMalloc(&d_vector, SIZE*sizeof(float));
        cudaMalloc(&d_result, sizeof(float));
        cudaMemcpy(d_vector, h_vector, SIZE*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(float));
        __START_BIS__
        sumall_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector, d_result);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP_BIS__(cudaKRun);
        cudaMemcpy(h_vector, d_vector, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vector);
        cudaFree(d_result);
        __NEXT__(cudaRun)
        
        h_result = 0.;
        for (unsigned int i(0) ; i!=SIZE ; i++)
            h_result += h_vector[i];
        __STOP__(nativeRun)
    }
    showMean(cudaRun);
    showMean(cudaKRun);
    showMean(nativeRun);
}    

