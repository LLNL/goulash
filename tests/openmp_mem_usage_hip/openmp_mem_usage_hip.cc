/*
  Copyright (c) 2019-20, Lawrence Livermore National Security, LLC. and other
  Goulash project contributors LLNL-CODE-795383, All rights reserved.
  For details about use and distribution, please read LICENSE and NOTICE from
  the Goulash project repository: http://github.com/llnl/goulash
  SPDX-License-Identifier: BSD-3-Clause
*/

/* Designed to measure OpenMP offloading runtime memory usage and overheads on
 * either the first GPU (-DSINGLE_GPU) or all GPUs with a compiler's runtime.
 * Some OpenMP overheads come just from linking to the library (GPU memory
 * allocated before main), so it is recommended to compile without OpenMP 
 * (-DNO_OPENMP) to get a good baseline for GPU free memory without OpenMP.
 * Also measures memory freed by OpenMP soft and hard pauses.
 *
 * Pulled code from LLNL's goulash test rush_larsen_openmp.cc (Rob Blake/LLNL)
 * and LLNL's CORAL1 test_sierra_node.cc (John Gyllenhaal/LLNL).  
 * Combined and ported to pure HIP by John Gyllenhaal (gyllenhaal1@llnl.gov)
 * at LLNL 12/10/20
 * 
 * Version History:
 * V1.1 12/18/20 fix host leaks, makes leak option since exposed issues
 * V1.0 12/10/20 initial release, found later to leak pinned host memory
 */
#include "hip/hip_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#ifndef NO_OPENMP
#include <omp.h>
#endif

/* Allow version to be printed in output */
static const char *version_string="Version 1.1 (12/18/20)";


/* Returns time in seconds (double) since the first call to secs_elapsed
 * (i.e., the first call returns 0.0).
 */
double secs_elapsed( void )
{
    static double base_time = -1;
    struct timeval ts;
    int status;
    double new_time;

    /* Get wall-clock time */
    /* status = getclock( CLOCK_REALTIME, &ts ); */
    status = gettimeofday( &ts, NULL );

    /* Return 0.0 on error */
    if( status != 0 ) return 0.0;

    /* Convert structure to double (in seconds ) (a large number) */
    new_time = (double)ts.tv_sec + (double)ts.tv_usec * 1e-6;

    /* If first time called, set base_time
     * Note: Lock shouldn't be needed, since even if multiple
     *       threads initialize this, it will be to basically
     *       the same value.
     */
    if (base_time < 0)
	base_time = new_time;

    /* Returned offset from first time called */
    return (new_time - base_time);
}

/* Works like printf, except prefixes wall-clock time (using secs_elapsed)
 * and writes to stdout.  Also flushes stdout, so messages stay
 * in reasonable order.   Now also prints diff;
 */
double last_printf_timestamp=0.0;

void printf_timestamp (const char * fmt, ...)
{
    va_list args;
    char buf[4096];

    /* Get wall-clock time since first call to secs_elapsed */
    double sec = secs_elapsed();
    double diff = sec - last_printf_timestamp;
    last_printf_timestamp=sec;

    /* Print out passed message to big buffer*/
    va_start (args, fmt);
    vsnprintf (buf, sizeof(buf), fmt, args);

    /* Print out timestamp and diff seconds with buffer*/
    fprintf (stdout, "%7.3f (%05.3fs): %s", sec, diff, buf);

    va_end (args);

    /* Flush stdout, so message appears immediately */
    fflush (stdout);
}

/* Prints to stderr (flushes stdout first) with timestamp and exits */
void punt (const char * fmt, ...)
{
    va_list args;
    char buf[4096];

    /* Get wall-clock time since first call to secs_elapsed */
    double sec = secs_elapsed();

    /* Flush stdout, so message appear in reasonable order */
    fflush (stdout);

    /* Print out passed message to big buffer*/
    va_start (args, fmt);
    vsnprintf (buf, sizeof(buf), fmt, args);

    /* Print out timestamp with buffer*/
    fprintf (stderr, "%8.3f ERROR: %s", sec, buf);

    va_end (args);
    exit(1);
}

/* Measures available GPU memory for gpu_id by mallocing data on the GPU
 * until fails, then frees memory in reverse order allocated.
 * Returns GBs free memory as a double.
 */
#define MAX_1GB_ALLOCS 2024
#define MAX_16MB_ALLOCS 1024
double last_gpu_mem=-1;
double diff_last_gpu_mem=-1;

double measure_gpu_mem (int gpu_id)
{
    hipError_t err;
    void *mem_1GB_array[MAX_1GB_ALLOCS];
    void *mem_16MB_array[MAX_16MB_ALLOCS];
    int index_1GB=0;
    int index_16MB=0;
    double alloced_mem = 0.0;

    err = hipSetDevice(gpu_id);
    if (err != hipSuccess)
    {
        punt ("GPU %i failed hipSetDevice: %s", gpu_id, hipGetErrorString(err));
    }

    /* Make sure GPU is not totally broken before doing test */
    double* dummyfirst;
    err=hipMalloc(&dummyfirst, sizeof(double)*4);
    if (err != hipSuccess)
    {
        punt ("GPU %i failed first hipMalloc: %s", gpu_id, hipGetErrorString(err));
    }
    err=hipFree(dummyfirst);
    if (err != hipSuccess)
    {
	punt ("GPU %i failed first hipFree: %s", gpu_id,  hipGetErrorString(err));
    }

    /* Allocate as many 1GB chunks as possible */
    while (index_1GB < MAX_1GB_ALLOCS)
    {
	err=hipMalloc(&mem_1GB_array[index_1GB], 1024*1024*1024);
	if (err != hipSuccess)
	{
	    /* Reset error so next malloc doesn't fail */
	    hipGetLastError();
	    break;
	}
	index_1GB+=1;
	alloced_mem +=1.0;
    }

    /* If this limit hit, likely need to up limit */
    if (index_1GB >= MAX_1GB_ALLOCS)
	printf_timestamp("WARNING: May have measured less then full gpu memory!   MAX_1GB_ALLOCS (%i) hit!\n", MAX_1GB_ALLOCS);
       
    /* Allocate in 16MB chunks until malloc fails */
    while (index_16MB < MAX_16MB_ALLOCS)
    {
	err=hipMalloc(&mem_16MB_array[index_16MB], 16*1024*1024);
	if (err != hipSuccess)
	{
	    /* Reset error so next malloc doesn't fail */
	    hipGetLastError();
	    break;
	}
	index_16MB+=1;
	alloced_mem += (16.0/1024.0);
    }

    /* Warn if limits hit or allocated more than 1GB in 16MB chunks */
    if ((index_16MB >= 128) && (index_1GB < MAX_1GB_ALLOCS))
	printf_timestamp("WARNING: GPU memory may be fragmented, more than 1GB of 16MB allocs (%4.2lf GBs) allocated! \n", ((index_16MB*16.0)/1024.0) );


    if (index_16MB >= MAX_16MB_ALLOCS)
	printf_timestamp("WARNING: May have measured less then full gpu memory!   MAX_16MB_ALLOCS (%i) hit!\n", MAX_16MB_ALLOCS);

    /* Free memory in reverse order allocated */
    for (int i=index_16MB-1; i >= 0; i--)
    {
	err=hipFree(mem_16MB_array[i]);
	if (err != hipSuccess)
	{
	    punt ("GPU %i failed hipFree 16MB alloc %i : %s", gpu_id, i, hipGetErrorString(err));
	}
    }

    for (int i=index_1GB-1; i >= 0; i--)
    {
	err=hipFree(mem_1GB_array[i]);
	if (err != hipSuccess)
	{
	    punt ("GPU %i failed hipFree 1GB alloc %i : %s", gpu_id, i, hipGetErrorString(err));
	}
    }

    /* Make it easier to print out diffs of memory measurements */
    diff_last_gpu_mem = alloced_mem-last_gpu_mem;
    last_gpu_mem = alloced_mem;

    return (alloced_mem);
}

/* If using OpenMP, do OpenMP version of rush_larsen kernel 1 time on specified
 * GPU using the nCells specified (derived from GBs desired on command line).
 *
 * This polynomial is a fit to the dynamics of a small part of a cardiac 
 * myocyte, specifically the fast sodium m-gate described here:
 * https://www.ncbi.nlm.nih.gov/pubmed/16565318
 */
#ifndef NO_OPENMP
void openmp_rush_larsen(int gpu_id, long nCells)
{
    double openmp_mem;
    double resultFromGPU = -1000000.0;
    double *m_gate, *Vm;

    omp_set_default_device(gpu_id);
    printf_timestamp("GPU %i: Time used omp_set_default_device\n", gpu_id);

    openmp_mem = measure_gpu_mem(gpu_id);
    printf_timestamp("GPU %i: Free memory   before rush_larsen: %6.2lfgb\n", gpu_id, openmp_mem);

#ifndef USE_CALLOC
    long buf_size=nCells*sizeof(double);
    hipHostMalloc(&m_gate, buf_size, 0);
    hipHostMalloc(&Vm, buf_size, 0);
    printf_timestamp("GPU %i: Time used hipHostMalloc\n", gpu_id);
    memset(m_gate, 0, buf_size);
    memset(Vm, 0, buf_size);
    printf_timestamp("GPU %i: Time used memset to 0\n", gpu_id);
#else
    m_gate = (double*)calloc(nCells,sizeof(double));
    Vm = (double*)calloc(nCells,sizeof(double));
    printf_timestamp("GPU %i: Time used calloc\n", gpu_id);
#endif

    openmp_mem = measure_gpu_mem(gpu_id);
#ifndef USE_CALLOC
    printf_timestamp("GPU %i: Free memory  after hipHostMalloc: %6.2lfgb (%6.2lfgb)\n", gpu_id, openmp_mem, diff_last_gpu_mem);
#else
    printf_timestamp("GPU %i: Free memory         after calloc: %6.2lfgb (%6.2lfgb)\n", gpu_id, openmp_mem, diff_last_gpu_mem);
#endif

#pragma omp target enter data map(to: m_gate[:nCells])
#pragma omp target enter data map(to: Vm[:nCells])
    printf_timestamp("GPU %i: Time used target enter data map\n", gpu_id);
    openmp_mem = measure_gpu_mem(gpu_id);
    printf_timestamp("GPU %i: Free memory after enter data map: %6.2lfgb (%6.2lfgb)\n", gpu_id, openmp_mem, diff_last_gpu_mem);

#pragma omp target teams distribute parallel for map(from:resultFromGPU)
    for (int ii=0; ii<nCells; ii++) {
	double sum1,sum2;
	const double x = Vm[ii];
	const int Mhu_l = 10;
	const int Mhu_m = 5;
	const double Mhu_a[] = { 9.9632117206253790e-01,  4.0825738726469545e-02,  6.3401613233199589e-04,  4.4158436861700431e-06,  1.1622058324043520e-08,  1.0000000000000000e+00,  4.0568375699663400e-02,  6.4216825832642788e-04,  4.2661664422410096e-06,  1.3559930396321903e-08, -1.3573468728873069e-11, -4.2594802366702580e-13,  7.6779952208246166e-15,  1.4260675804433780e-16, -2.6656212072499249e-18};

	sum1 = 0;
	for (int j = Mhu_m-1; j >= 0; j--)
	    sum1 = Mhu_a[j] + x*sum1;

	sum2 = 0;
	int k = Mhu_m + Mhu_l - 1;
	for (int j = k; j >= Mhu_m; j--)
	    sum2 = Mhu_a[j] + x * sum2;
	double mhu = sum1/sum2;

	const int Tau_m = 18;
	const double Tau_a[] = {1.7765862602413648e+01*0.02,  5.0010202770602419e-02*0.02, -7.8002064070783474e-04*0.02, -6.9399661775931530e-05*0.02,  1.6936588308244311e-06*0.02,  5.4629017090963798e-07*0.02, -1.3805420990037933e-08*0.02, -8.0678945216155694e-10*0.02,  1.6209833004622630e-11*0.02,  6.5130101230170358e-13*0.02, -6.9931705949674988e-15*0.02, -3.1161210504114690e-16*0.02,  5.0166191902609083e-19*0.02,  7.8608831661430381e-20*0.02,  4.3936315597226053e-22*0.02, -7.0535966258003289e-24*0.02, -9.0473475495087118e-26*0.02, -2.9878427692323621e-28*0.02,  1.0000000000000000e+00};

	sum1 = 0;
	for (int j = Tau_m-1; j >= 0; j--)
	    sum1 = Tau_a[j] + x*sum1;

	double tauR = sum1;

	m_gate[ii] += (mhu - m_gate[ii])*(1-exp(-tauR));

        /* Make sure actually running on GPU and use calc result 
         * so whole loop unlikely to be optimized away
         */
        if (ii == 0)
        {
#if 0
                  resultFromGPU=m_gate[ii]+1000000.0;
#else
             /* Only set result value to 1000000.297949 if on GPU */
             if (omp_is_initial_device() == 0)
                  resultFromGPU=m_gate[ii]+1000000.0;
             else
                  resultFromGPU=m_gate[ii]+2000000.0;
#endif
 
        }
    }

    /* Make sure really got back result from GPU */
    if (resultFromGPU < -999999)
        printf ("WARNING: Rush Larsen OpenMP kernel did not copy back data from GPU! (result %lf)\n", resultFromGPU);
 
    /* Did we get an unexpected omp_is_initial_device() results? */
    else if ((resultFromGPU > 2000000.297948) && (resultFromGPU < 2000000.297950))
        printf ("WARNING: Rush Larsen OpenMP kernel indicated omp_is_initial_device() returned != 0 (%lf != 1000000.297949)\n", resultFromGPU);

    /* Make sure have reasonable answer returned from kernel */
    else if ((resultFromGPU < 1000000.297948) || (resultFromGPU > 1000000.297950))
        printf ("WARNING: Rush Larsen OpenMP kernel mapped back unexpected answer (%lf != 1000000.297949).  Bad result!\n", resultFromGPU);
  
    printf_timestamp("GPU %i: Time used just OpenMP Kernel\n", gpu_id);
    openmp_mem = measure_gpu_mem(gpu_id);
    printf_timestamp("GPU %i: Free memory  after OpenMP kernel: %6.2lfgb (%6.2lfgb)\n", gpu_id, openmp_mem, diff_last_gpu_mem);
#pragma omp target exit data map(delete:m_gate[:nCells])
#pragma omp target exit data map(delete:Vm[:nCells])
    openmp_mem = measure_gpu_mem(gpu_id);
    printf_timestamp("GPU %i: Free memory  after exit data map: %6.2lfgb (%6.2lfgb)\n", gpu_id, openmp_mem, diff_last_gpu_mem);


    /* Free memory unless intentially leaking to see issues caused by user leaks */
    /* V1.0 accidently didn't free memory and exposed unexpected crashes and hangs */
#ifndef USE_CALLOC
#ifndef LEAK_HOST_MEM
    hipHostFree(m_gate);
    hipHostFree(Vm);
    openmp_mem = measure_gpu_mem(gpu_id);
    printf_timestamp("GPU %i: Free memory    after hipHostFree: %6.2lfgb (%6.2lfgb)\n", gpu_id, openmp_mem, diff_last_gpu_mem);
#else
    printf_timestamp("GPU %i: Leaking host memory - not calling hipHostFree for hipHostMalloced arrays.\n", gpu_id);
#endif
#else
#ifndef LEAK_HOST_MEM
    free(m_gate);
    free(Vm);
    openmp_mem = measure_gpu_mem(gpu_id);
    printf_timestamp("GPU %i: Free memory     after array free: %6.2lfgb (%6.2lfgb)\n", gpu_id, openmp_mem, diff_last_gpu_mem);
#else
    printf_timestamp("GPU %i: Leaking host memory - not calling free for calloced arrays.\n", gpu_id);h
#endif
#endif

}
#endif

/* Parse arguments and run tests */
int main(int argc, char* argv[]) 
{
    hipError_t err;
    int num_gpus=-1;
    long max_iterations=1, num_cells=200;
    double GPU_mem_used=0.0;
    double hip_mem, openmp_mem, soft_pause_mem, hard_pause_mem;

    if (argc != 3)
    {
	printf ("Usage: %s  Iterations  Kernel_GBs_used\n", argv[0]);
	printf ("\n");
	printf ("Measure fixed overheads: %s   1 .0000001\n", argv[0]);
	printf ("  Measure all overheads: %s   1 31\n", argv[0]);
	printf (" Runtime stability test: %s 100 31\n", argv[0]);
	printf ("\n");
	printf ("%s\n", version_string);
	printf ("\n");
#ifdef NO_OPENMP
	printf ("Measures GPU memory available without OpenMP runtimes\n");
        printf ("No kernels launched! Kernel_GBs_used ignored! Compiled with -DNO_OPENMP)\n");
#else
	printf ("Measures GPU memory available with OpenMP offloading before/after kernel launch,\n");
        printf ("data map, and omp_soft_pause and omp_hard_pause (Compiled without -DNO_OPENMP)\n");
#endif
	printf ("\n");
#ifdef USE_CALLOC
	printf ("Uses calloc to allocate unpinned bufs   (Compiled with -DUSE_CALLOC)\n");
#ifdef LEAK_HOST_MEM
	printf ("All unpinned calloc bufs will be leaked (Compiled with -DLEAK_HOST_MEM)\n");
#else
	printf ("All unpinned calloc bufs will be freed  (Compiled without -DLEAK_HOST_MEM)\n");
#endif
#else
	printf ("Uses hipHostMalloc to allocate pinned bufs   (Compiled without -DUSE_CALLOC)\n");
#ifdef LEAK_HOST_MEM
	printf ("All pinned hipHostMalloc bufs will be leaked (Compiled with -DLEAK_HOST_MEM)\n");
#else
	printf ("All pinned hipHostMalloc bufs will be freed  (Compiled without -DLEAK_HOST_MEM)\n");
#endif
#endif
	printf ("\n");
#ifdef SINGLE_GPU
	printf ("Only GPU 0 will be tested (Compiled with -DSINGLE_GPU)\n");
#else
	printf ("All GPUs will be tested (Compiled without -DSINGLE_GPU)\n");
#endif
	printf ("\n");
	printf ("Questions? Contact John Gyllenhaal (gyllenhaal1@llnl.gov)\n");
	exit (1);
    }
   
    /* Get iteration count and target GPU memory used arguments */
    max_iterations = atol(argv[1]);
    GPU_mem_used=atof(argv[2]);

    /* Calculate num_cells from target memory target */
    num_cells = (long) ((GPU_mem_used * 1024.0 * 1024.0 * 1024.0) / (sizeof(double) * 2));
    /* Must have at least 1 cell */
    if (num_cells < 1)
	num_cells = 1;

    err=hipGetDeviceCount(&num_gpus);
    if (err != hipSuccess)
    {
        punt ("Failed during hipGetDeviceCount: %s",  hipGetErrorString(err));
    }

    /* Optionally restrict test to GPU 0 */
#ifdef SINGLE_GPU
    num_gpus=1;
#endif

#ifdef NO_OPENMP
    printf("%s: Measuring Memory on the %i GPUs available with no OpenMP GPU enabled\n", version_string, num_gpus);
    printf("Iterations: %ld\n", max_iterations);
#else
    printf("%s: Measuring memory free on %i GPUs with OpenMP GPU kernels and pauses\n", version_string, num_gpus);
    printf("Iterations: %ld  Rush Larsen GPU kernel mem: %.8lf GBs  Cells: %ld\n", max_iterations, (num_cells * sizeof(double) * 2.0/(1024.0*1024.0*1024.0)), num_cells);
#endif

    printf ("Compiler flags:");
#ifdef SINGLE_GPU
    printf (" -DSINGLE_GPU");
#else
    printf (" -USINGLE_GPU");
#endif
#ifdef NO_OPENMP
    printf (" -DNO_OPENMP");
#else
    printf (" -UNO_OPENMP");
#endif
#ifdef USE_CALLOC
    printf (" -DUSE_CALLOC");
#else
    printf (" -UUSE_CALLOC");
#endif
#ifdef LEAK_HOST_MEM
    printf (" -DLEAK_HOST_MEM");
#else
    printf (" -ULEAK_HOST_MEM");
#endif
    printf("\n");
#ifdef SINGLE_GPU
    printf ("Due to -DSINGLE_GPU, will only test GPU 0\n");
#endif
#ifdef NO_OPENMP
    printf ("Due to -DNO_OPENMP, only free memory without OpenMP runtimes loaded measured.\n");
#endif
#ifdef USE_CALLOC
    printf ("Due to -DUSE_CALLOC, calloc used for mapped buffers instead of hipHostMalloc + memset(0).\n");
#ifdef LEAK_HOST_MEM
    printf ("Due to -DLEAK_HOST_MEM, all host buffers allocated with calloc will be leaked.\n");
#endif
#else
#ifdef LEAK_HOST_MEM
    printf ("Due to -DLEAK_HOST_MEM, all host buffers allocated with hipHostMalloc will be leaked.\n");
#endif
#endif
    fflush(stdout);

    for (int iteration=0; iteration < max_iterations; iteration++)
    {
        printf("\n-----------\n\n");
        printf_timestamp("Iteration %i of %i\n", iteration+1, max_iterations);

	for (int gpu_id=0; gpu_id < num_gpus; gpu_id++)
	{
	    printf("\n");

#ifdef NO_OPENMP
	    /* Get available gpu memory with no HIP kernels and no OpenMP offloading */
	    hip_mem = measure_gpu_mem(gpu_id);
	    printf_timestamp("GPU %i: Free memory with NO OpenMP runtime: %6.2lfgb\n", gpu_id, hip_mem);

#else
            /* Get available gpu memory before and after running kernel and soft/hard pauses */ 
            /* Run OpenMP offloading kernel with tiny data set */
	    openmp_rush_larsen(gpu_id, num_cells);

            /* Do OpenMP soft pause */
	    omp_pause_resource(omp_pause_soft, gpu_id);
	    printf_timestamp("GPU %i: Time used omp_pause_soft\n", gpu_id);
	    soft_pause_mem = measure_gpu_mem(gpu_id);
	    printf_timestamp("GPU %i: Free memory after omp_pause_soft: %6.2lfgb (%6.2lfgb)\n", gpu_id, soft_pause_mem, diff_last_gpu_mem);

            /* DO OpenMP hard pause */
	    omp_pause_resource(omp_pause_hard, gpu_id);
	    printf_timestamp("GPU %i: Time used omp_pause_hard\n", gpu_id);
	    hard_pause_mem = measure_gpu_mem(gpu_id);
	    printf_timestamp("GPU %i: Free memory after omp_pause_hard: %6.2lfgb (%6.2lfgb)\n", gpu_id, hard_pause_mem, diff_last_gpu_mem);
#endif
	}
    }
  

    return 0;
}

