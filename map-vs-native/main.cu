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
    std::cout << 1000.*sum/v.size() << " microseconds" << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "SIZE (vs. Native): " << SIZE << std::endl;
    cudaEvent_t start, stop, startBis, stopBis;
    std::vector<float> cRun, ckRun, hRun;
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
        cudaMemcpy(d_vector, h_vector, SIZE*sizeof(float), cudaMemcpyHostToDevice);
        __START_BIS__
        square_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP_BIS__(ckRun)
        cudaMemcpy(h_vector, d_vector, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vector);
        __NEXT__(cRun)
        for (unsigned int i(0) ; i!=SIZE ; i++)
            h_vector[i] = h_vector[i]*h_vector[i];
        __STOP__(hRun)
    }
    showMean(cRun);
    showMean(ckRun);
    showMean(hRun);
}    

