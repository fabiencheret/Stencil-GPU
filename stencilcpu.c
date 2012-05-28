#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <CL/opencl.h>
#include <pthread.h>
#include <omp.h>

#include "constantes.h"

#define TIME_DIFF(t1, t2) \
    ((t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec))

#define MAX_DEVICES 5


#define error(...) do { fprintf(stderr, "Error: " __VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#define check(err, ...)					\
    do {							\
    if(err != CL_SUCCESS) {				\
    fprintf(stderr, "(%d) Error: " __VA_ARGS__, err);	\
    exit(EXIT_FAILURE);				\
    }							\
    } while(0)
//
//float optimizer;
//struct timeval tcpu,tgpu;
//unsigned int ydim_gpu = YDIM_GPU;
//

/* Version CPU multicoeur */
void stencil_multi(float* B, const float* A, int ydim)
{
#pragma omp parallel for schedule(guided)
    for(int y=0; y<ydim; y++)
        for(int x=0; x<XDIM; x++)
            B[y*LINESIZE + x] = 0.75*A[y*LINESIZE + x] +
                                0.25*( A[y*LINESIZE + x - 1] + A[y*LINESIZE + x + 1] +
                                       A[(y-1)*LINESIZE + x] + A[(y+1)*LINESIZE + x]);
}

/* Version CPU pour comparaison */
void stencil(float* B, const float* A, int ydim)
{
    for(int y=0; y<ydim; y++)
        for(int x=0; x<XDIM; x++)
            B[y*LINESIZE + x] = 0.75*A[y*LINESIZE + x] +
                                0.25*( A[y*LINESIZE + x - 1] + A[y*LINESIZE + x + 1] +
                                       A[(y-1)*LINESIZE + x] + A[(y+1)*LINESIZE + x]);
}

struct timeval temps1, temps2;


/* fonction appelée par le thread dès sa création */
void* calcul_cpu(void* p)
{
    struct double_matrice* container = (struct double_matrice*) p;
    stencil_multi(container->out, container->in, container->ydim_cpu);
    gettimeofday(&temps1,NULL);
    return NULL;
}

int ydim_gpu = 512;

int main(int argc, char** argv)
{

    int numIterations = NB_ITER;

    if(argc > 0)
    {
        numIterations = atoi(argv[1]);
        ydim_gpu = atoi(argv[2]);
    }

    const unsigned int line_size = LINESIZE;
    const unsigned int mem_size= TOTALSIZE*sizeof(float);
    const unsigned int mem_size_gpu = SIZE_GPU * sizeof(float);

    float *h_idata = NULL;
    float *h_odata = NULL;
    struct double_matrice container;


    struct timeval tv1,tv2,tcpu1,tcpu2;

    // Allocation of input & output matrices
    //
    h_idata = malloc(mem_size);
    h_odata = malloc(mem_size);

    container.in  = h_idata + LINESIZE * (YDIM_GPU) + OFFSET;
    container.out = h_odata + LINESIZE * (YDIM_GPU) + OFFSET;
    container.ydim_cpu = YDIM_CPU;



    // Initialization of input & output matrices
    //
    srand(1234);
    for(unsigned int i = 0; i < TOTALSIZE; i++)
    {
        h_idata[i]=rand();
        h_odata[i]=0.0;
    }



    /* Version cpu pour comparaison */

    void * tmp_switch;


    float* reference = (float*) malloc(mem_size);
    float* reference_i = (float*) malloc(mem_size);
    for(unsigned int i = 0; i < TOTALSIZE; i++)
    {
        reference[i]   = 0.0;
        reference_i[i] = h_idata[i];
    }

    gettimeofday(&tcpu1,NULL);

    for(int i=0; i<numIterations; ++i)
    {
        stencil(reference + OFFSET, reference_i + OFFSET, YDIM);
        tmp_switch  = reference;
        reference   = reference_i;
        reference_i = tmp_switch;
    }

    if(numIterations%2)
    {
        tmp_switch  = reference;
        reference   = reference_i;
        reference_i = tmp_switch;
    }
    gettimeofday(&tcpu2,NULL);

    float timecpu=((float)TIME_DIFF(tcpu1,tcpu2)) / 1000;



    pthread_t thread;

    printf("nombre d'itérations: %d\n",numIterations);

    gettimeofday(&tv1, NULL);

    for(int i = 0; i<numIterations; i++) // Iterations are done inside the kernel
    {
        stencil_multi(container.out,container.in,container.ydim_cpu);

        tmp_switch = container.out;
        container.out = container.in;
        container.in = tmp_switch;

    }
    gettimeofday(&tv2, NULL);

    tmp_switch = d_odata;
    d_odata = d_idata;
    d_idata = tmp_switch;



    float time1=((float)TIME_DIFF(tv1,tv2)) / 1000;
    // Read back the results from the device to verify the output
    //

    printf("%f\t%f ms (%fGo/s)\t%f ms (%fGo/s)\n", timecpu/time1,
           time1, numIterations * 3*mem_size / time1 / 1000000,
           timecpu, numIterations * 3*mem_size / timecpu / 1000000);

    // Validate our results
    //
    unsigned int errors=0;

    float * h_fdata = h_odata;

    for(unsigned int i=0; i<TOTALSIZE; i++)
    {
        if((reference[i]-h_fdata[i])/reference[i] > 1e-6)
        {
            if(errors < 10) printf(" %u %f vs %f\n", i, h_fdata[i], reference[i]);
            errors++;
        }
    }
    if(errors)
        fprintf(stderr,"%d erreurs !\n", errors);
    else
        fprintf(stderr,"pas d'erreurs, cool !\n");
    free(reference_i);
    free(reference);

    // Shutdown and cleanup
    //
    free(h_odata);
    free(h_idata);

    return 0;
}

