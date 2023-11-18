//
// Created by ahmed on 18/11/23.
//

#ifndef LTFAT_LTFAT_CPU_H
#define LTFAT_LTFAT_CPU_H
#include <iostream>


void comp_idgt_fb(float*** coef,float* g,int L,int a,int M,int W,int gl,float* ff, float** f){

    int glh=gl/2;
    int glh_d_a = std::ceil((glh * 1.0) / (a));


    for(int w = 0 ; w<W ; w++){

        for(int n = glh_d_a; n < (L - (gl + 1) / 2) / a + 1 ; n++){

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
    }

}





#endif //LTFAT_LTFAT_CPU_H
