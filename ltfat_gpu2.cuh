//
// Created by ahmed on 18/11/23.
//

#ifndef LTFAT_LTFAT_GPU2_CUH
#define LTFAT_LTFAT_GPU2_CUH
#include <iostream>


/*
this kernel corresponds to this part:

    for ii=0:gl/M-1
      for m=0:delay-1
        ff(m+ii*M)=coef(M-delay+m,n,w)*g(m+ii*M);
      end;
 */
__global__
void ff_update1(float*** coef, float* ff, float* g, int M, int n,int w, int delay, int gl){

    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (ii >= gl/M)return;
    if(m>=delay-1)return;

    ff[m+ii*m] = coef[M-delay-m][n][w] * g[m+ii*M];
}

/*
 this kernel corresponds to this part

    for ii=0:gl/M-1
      for m=0:M-delay-1
        ff(m+ii*M+delay)=coef(m,n,w)*g(m+delay+ii*M);
      end;
 */
__global__
void ff_update2(float*** coef, float* ff, float* g, int M, int n,int w, int delay, int gl){

    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (ii >= gl/M)return;
    if(m>=M - delay-1)return;

    ff[m+ii*M + delay] = coef[m][n][w] * g[m+delay+ii*M];
}

/*
 this kernel corresponds to this part

    for ii=0:ep-sp
      f(ii+sp+1,w)=f(ii+sp+1,w)+ff(ii+1);
    end;
 */
__global__
void f_update(float*ff, float** f,int w, int sp, int ep){

    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if (ii >= ep - sp)return;
    f[ii + sp][w] += ff[ii];

}

// main kernel handling the middle case
__global__
void comp_idgt_fb_2_kernel(float*** coef,float* g,int L,int a,int M,int W,int gl,float* ff, float** f,int glh, int glh_d_a) {

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    int n_start = glh_d_a;
    int n_end = (L - (gl + 1) / 2) / a + 1;
    n+=n_start;

    if (w >= W)return;
    if (n >= n_end )return;

    int delay = -n*a+glh % M;

    dim3 threadsPerBlock(256,256);

    int blocksPerGridX= ((gl/M) + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksPerGridY= ((delay-1) + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 blocksPerGrid(blocksPerGridX,blocksPerGridY);

    ff_update1<<<blocksPerGrid, threadsPerBlock>>>(coef, ff, g, M, n, w, delay, gl);

    blocksPerGridY= ((M-delay-1) + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 blocksPerGrid2(blocksPerGridX,blocksPerGridY);

    ff_update2<<<blocksPerGrid2, threadsPerBlock>>>(coef, ff, g, M, n, w, delay, gl);


    int sp = (n * a - glh) % L;
    int ep = (n * a - glh + gl-1) % L;

    blocksPerGridX= ((ep - sp) + threadsPerBlock.x - 1) / threadsPerBlock.x;
    f_update<<<blocksPerGridX, threadsPerBlock.x>>>(ff,f,w,sp,ep);

}

// host function to lunch device kernel
void comp_idgt_fb_2_gpu(float*** coef,float* g,int L,int a,int M,int W,int gl,float* ff, float** f) {

    int glh=gl/2;
    int glh_d_a = std::ceil((glh * 1.0) / (a));

    int n_start = glh_d_a;
    int n_end = (L - (gl + 1) / 2) / a + 1;

    int N_RANGE = n_end - n_start + 1;

    int threadsPerBlock = 256;
    int blocksPerGridX= (W + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridY= (N_RANGE + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocksPerGrid(blocksPerGridX,blocksPerGridY);
    dim3 threadsPerBlock2D(256,256);


    float*** coef_d; //
    float* g_d;
    float* ff_d;
    float** f_d;
    //TODO allocate gpu memory and copy data from host to device
    comp_idgt_fb_2_kernel<<<blocksPerGrid,threadsPerBlock2D>>>(coef_d, g_d, L, a, M, W, gl, ff_d, f_d, glh, glh_d_a);

    // TODO copy data from device back to host

}
#endif //LTFAT_LTFAT_GPU2_CUH
