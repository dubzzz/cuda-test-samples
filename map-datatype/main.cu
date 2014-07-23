#include <iostream>
#include <stdio.h>
#include <vector>

#define MAX_THREADS 256
#define SIZE 131072

#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define __STOP__(_V) cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop); _V.push_back(time); cudaEventDestroy(start); cudaEventDestroy(stop);
#define __NEXT__(_V) __STOP__(_V) __START__

template<class T>
__global__ void square_kernel(T *d_vector)
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
    std::cout << "SIZE (Datatype): " << SIZE << std::endl;
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
    
    for (unsigned int i(0) ; i!=1000 ; i++)
    {
        int *d_vector_i;
        float *d_vector_f;
        double *d_vector_d;
        
        /* INT */
        cudaMalloc(&d_vector_i, SIZE*sizeof(int));
        cudaMemcpy(d_vector_i, h_vector_i, SIZE*sizeof(int), cudaMemcpyHostToDevice);
        __START__
        square_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector_i);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP__(intRun);
        cudaMemcpy(h_vector_i, d_vector_i, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_vector_i);
        
        /* FLOAT */
        cudaMalloc(&d_vector_f, SIZE*sizeof(float));
        cudaMemcpy(d_vector_f, h_vector_f, SIZE*sizeof(float), cudaMemcpyHostToDevice);
        __START__
        square_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector_f);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP__(floatRun);
        cudaMemcpy(h_vector_f, d_vector_f, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vector_f);
        
        /* DOUBLE */
        cudaMalloc(&d_vector_d, SIZE*sizeof(double));
        cudaMemcpy(d_vector_d, h_vector_d, SIZE*sizeof(double), cudaMemcpyHostToDevice);
        __START__
        square_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector_d);
        cudaThreadSynchronize(); // Block until the device is finished
        __STOP__(doubleRun);
        cudaMemcpy(h_vector_d, d_vector_d, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_vector_d);
    }
    showMean(intRun);
    showMean(floatRun);
    showMean(doubleRun);
}    

