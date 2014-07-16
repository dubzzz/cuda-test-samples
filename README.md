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

