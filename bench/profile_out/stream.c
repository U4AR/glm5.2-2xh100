
/* Two kernels:
     read  -- pure streaming read (8 B/elem).  This is the right ceiling for
              MoE weight streaming, which never writes the weights back.
     triad -- classic STREAM triad. Counted at 32 B/elem, not 24: the store to
              c[] pulls the line in first (write-allocate / RFO), so the DRAM
              actually moves 4 words per element, not 3.
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char** argv){
    size_t N = (argc>1)? (size_t)atoll(argv[1]) : (size_t)200000000;
    int iters = (argc>2)? atoi(argv[2]) : 8;
    double *a=aligned_alloc(64,N*8),*b=aligned_alloc(64,N*8),*c=aligned_alloc(64,N*8);
    #pragma omp parallel for
    for(size_t i=0;i<N;i++){a[i]=1.0;b[i]=2.0;c[i]=0.0;}
    double best_t=0, best_r=0;
    for(int it=0; it<iters; it++){
        double t0=omp_get_wtime();
        #pragma omp parallel for
        for(size_t i=0;i<N;i++) c[i]=a[i]+3.0*b[i];
        double dt=omp_get_wtime()-t0;
        double gbs = 32.0*(double)N/dt/1e9;
        if(gbs>best_t) best_t=gbs;

        /* integer accumulate: FP '+' is not associative so gcc refuses to
           vectorise a double reduction without -ffast-math, which would make
           this latency-bound instead of bandwidth-bound. */
        unsigned long long s=0; unsigned long long *ai=(unsigned long long*)a;
        t0=omp_get_wtime();
        #pragma omp parallel for reduction(+:s)
        for(size_t i=0;i<N;i++) s+=ai[i];
        dt=omp_get_wtime()-t0;
        gbs = 8.0*(double)N/dt/1e9;
        if(gbs>best_r) best_r=gbs;
        if(s==42) printf("x");   /* keep the reduction alive */
    }
    printf("%.1f %.1f\n", best_t, best_r);
    return 0;
}
