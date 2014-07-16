#Measurements on CUDA

##Configuration

Implementing a source code using CUDA is a real challenge. It requires to know how CUDA manages its memory and which kind of operations can be accelerated using CUDA instead of native-C. Size matters when dealing with a CUDA implementation: the larger the better.

CUDA can be a blessing for your runtime but it can also be worse than a native implementation. Indeed, a naive implementation can be very inefficient.

The following measurements have been made using the configuration:
+ Processor: Intel(R) Xeon(R) CPU E5606 @ 2.13GHz
+ GPU: GeForce GTX 780 – Compute capability = 3.5
+ Memory: 8GB
+ Operating system: Ubuntu 13.04
+ Kernel: Linux version 3.8.0-35-generic
+ CUDA compiler (nvcc): 5.5.0
+ C/C++ compiler (gcc/g++): 4.7.3

These measures are the mean of 1,000 measurements. They are expressed in milliseconds.

It was compiled and run using these instructions:
```bash
user@host:~$ /usr/local/cuda/bin/nvcc -m64  -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -po maxrregcount=16 -I/usr/local/cuda/include -I. -I.. -I../../common/inc -I/usr/local/cuda/samples/common/inc -o main.o -c main.cu
user@host:~$ g++ -m64 -o a.out main.o -L/usr/local/cuda/lib64 –lcudart
user@host:~$ ./a.out > /dev/null ; ./a.out > /dev/null ; ./a.out
```

The first run of the program is always slower than the others. For that reason, measures are based on the 3rd run.

##Impact of memory allocation

Memory allocation in CUDA can be very expensive. The following code has been used in order to measure the cost of allocation compared to kernel execution and copies back and forth.

Example of Map algorithm using CUDA:
```cuda
#include <iostream>

#define MAX_THREADS 256
#define SIZE 256

__global__ void square_kernel(float *d_vector)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= SIZE)
    return;
  d_vector[i] = d_vector[i]*d_vector[i];
}

int main(int argc, char **argv)
{
  cudaFree(0); // Force runtime API context establishment
  
  float h_vector[SIZE]; // For input and output
  for (unsigned int i(0) ; i!=SIZE ; i++)
    h_vector[i] = i;
  float *d_vector;
  
  cudaMalloc(&d_vector, SIZE*sizeof(float));
  cudaMemcpy(d_vector, h_vector, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  square_kernel<<<(SIZE+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(d_vector);
  cudaThreadSynchronize(); // Block until the device is finished
  cudaMemcpy(h_vector, d_vector, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_vector);
}
```

Time consumed line by line:

Vector size | cudaMalloc | cudaMemcpy (GPU>CPU) | kernel | cudaMemcpy (CPU>GPU) | cudaFree
------------|------------|----------------------|--------|----------------------|---------
32	| 144.31 	| 11.15 	| 22.50 	| 17.31 	| 108.84 
128	| 143.79 	| 11.21 	| 22.48 	| 17.69 	| 108.23 
512	| 144.60 	| 11.77 	| 22.67 	| 17.95 	| 110.30 
2 048	| 144.23 |	 13.72 |	 22.44 |	 20.68 |	 110.46 
8 192	| 142.33 	| 22.44 |	 22.30 |	 31.29 	| 111.00 
32 768	| 145.53 |	 65.15 |	 24.06 	| 73.02 |	 114.70 
131 072	| 144.03 |	 206.38 |	 26.17 |	 240.76 |	 115.45 
524 288	| 145.37 |	 685.64 |	 45.03 |	 783.88 |	 120.28 

For small vectors, the most significant parts are `cudaMalloc` and `cudaFree`. The time consumed by these two operations is constant – independent of vector size. Increasing the vector size makes these parts irrelevant. As shown on next graph the most significant piece of code in terms of time consumed becomes `cudaMemcpy` as the size increase. Kernel time also increases but very slowly compared to `cudaMemcpy`.
