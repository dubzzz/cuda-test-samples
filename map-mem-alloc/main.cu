#include <iostream>
#include <stdio.h>
#include <vector>

#define MAX_THREADS 256
#define SIZE 524288

#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define __STOP__(_V) cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop); _V.push_back(time); cudaEventDestroy(start); cudaEventDestroy(stop);
#define __NEXT__(_V) __STOP__(_V) __START__

__global__ void square_kernel(float *d_vector)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= SIZE)
        return;
    d_vector[i] = d_vector[i]*d_vector[i];
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
    std::cout << "SIZE (Memory Allocation): " << SIZE << std::endl;
    cudaEvent_t start, stop;
    std::vector<float> cMalloc, cMemcpy1, cKernel, cMemcpy2, cFree;
    float time(0);
    
    cudaFree(0); // Force runtime API context establishment
    float h_vector[SIZE]; // For input and output
    for (unsigned int i(0) ; i!=SIZE ; i++)
        h_vector[i] = i;
    
    for (unsigned int i(0) ; i!=1000 ; i++)
    {
        float *d_vector;
        __START__
        cudaMalloc(&d_vector, SIZE*sizeof(float));
        __NEXT__(cMalloc);
        cudaMemcpy(d_vector, h_vector, SIZE*sizeof(float), cudaMemcpyHostToDevice);
        __NEXT__(cMemcpy1);
        square_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector);
        cudaThreadSynchronize(); // Block until the device is finished
        __NEXT__(cKernel);
        cudaMemcpy(h_vector, d_vector, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        __NEXT__(cMemcpy2);
        cudaFree(d_vector);
        __STOP__(cFree);
    }
    showMean(cMalloc);
    showMean(cMemcpy1);
    showMean(cKernel);
    showMean(cMemcpy2);
    showMean(cFree);
}    

