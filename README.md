# LTFAT in CUDA
this is the implementation of the middle case of the **comp_idgt_fb** funciton inside from **LTFAT matlab toolbox**.

## description

the repo containes 3 header files: 
* ltfat_cpu.h (cpu implementation)
* ltfat_gpu1.cuh (first cuda implementation)
* ltfat_gpu2.cuh (second cuda implementation)


### ltfat_cpu.h
conatains cpu implementation of the algorithm (was used as to ease the implementation in cuda).
###  ltfat_gpu1.cuh
* in this implementation only the first 2 loops where parallelized.   
* the reset of the code was left to run sequentially.
* there is also a host function to launch the kernel.

###  ltfat_gpu2.cuh
* in this implementation the more paralelizem was introduced.
* same as the first implementation the main kernel is lunched to parallelize the first 2 loops.
the sum part was modified from :
``` matlab
    for ii=0:gl/M-1
      for m=0:delay-1
        ff(m+ii*M)=coef(M-delay+m,n,w)*g(m+ii*M);
      end;
      for m=0:M-delay-1
        ff(m+ii*M+delay)=coef(m,n,w)*g(m+delay+ii*M);
      end;
```
to
``` matlab
    for ii=0:gl/M-1
      for m=0:delay-1
        ff(m+ii*M)=coef(M-delay+m,n,w)*g(m+ii*M);
      end;
    for ii=0:gl/M-1
      for m=0:M-delay-1
        ff(m+ii*M+delay)=coef(m,n,w)*g(m+delay+ii*M);
      end;
```
* then 2 kernels were implemented to optimes the each of the 2 nested loops.
* then a kernel that implement this part is called
``` matlab
    for ii=0:ep-sp
      f(ii+sp+1,w)=f(ii+sp+1,w)+ff(ii+1);
    end;
```
**NOTE:** this implementation requires compute capability of 3.5 or higher as it uses CUDA Dynamic Parallelism.

## requirements
* cuda
* gpu with compute capability of 3.5 or higher
* cmake

## building 
* clone the repo and run the following:
```bash
mkdir build
cd build
cmake ..
make 
```
## Limitations
there are some limitations in these implementations as follows:
* the multi dimission arraies are implemented as pointer to pointers, but it's probably better to flatten them to a 1d array. 
* the allocation of the gpu memory copy the data from host to device and vice versa is missing the from host functions as there is 
* no testing or benchmarking was done as this is a prtial part from the algorithm and its not very clear what the input should look like.
