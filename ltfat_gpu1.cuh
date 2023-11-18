//
// Created by ahmed on 18/11/23.
//

#ifndef LTFAT_LTFAT_GPU1_CUH
#define LTFAT_LTFAT_GPU1_CUH


#include <iostream>
// main kernel handling the middle case

__global__
void comp_idgt_fb_1_kernel(float*** coef,float* g,int L,int a,int M,int W,int gl,float* ff, float** f,int glh, int glh_d_a) {

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    int n_start = glh_d_a;
    int n_end = (L - (gl + 1) / 2) / a + 1;
    n+=n_start;

    if (w >= W)return;
    if (n >= n_end )return;

    int delay= -n*a+glh % M;

    for (int ii = 0;ii<gl/M;ii++){

        for (int m=0;m<delay-1;m++){
            ff[m+ii*M] = coef[M-delay-m][n][w] * g[m+ii*M];
        }

        for(int m=0; m<M-delay-1;m++){
            ff[m+ii*M+delay] = coef[m][n][w] * g[m+delay+ii*M];
        }
    }

    int sp = (n * a - glh) % L;
    int ep = (n * a - glh + gl-1) % L;

    for (int ii = 0; ii<ep-sp; ii++){
        f[ii + sp][w] += ff[ii];
    }
}


// host function to lunch device kernel
void comp_idgt_fb_1_gpu(float*** coef,float* g,int L,int a,int M,int W,int gl,float* ff, float** f) {

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
    comp_idgt_fb_1_kernel<<<blocksPerGrid,threadsPerBlock2D>>>(coef_d, g_d, L, a, M, W, gl, ff_d, f_d, glh, glh_d_a);

    // TODO copy data from device back to host

}
#endif //LTFAT_LTFAT_GPU1_CUH
