/*
  Copyright (c) 2019-21, Lawrence Livermore National Security, LLC. and other
  Goulash project contributors LLNL-CODE-795383, All rights reserved.
  For details about use and distribution, please read LICENSE and NOTICE from
  the Goulash project repository: http://github.com/llnl/goulash
  SPDX-License-Identifier: BSD-3-Clause
*/

/* Designed to allow performance comparisons between naively written
 * HIP/CUDA and OpenMP GPU offloading. The OpenMP GPU offloading 
 * equivalent version (rush_larsen_omp.cc) should be able to match 
 * this HIP's versions performance.
 *
 * May be compiled with or without a COMPILERID specified (put in benchmark output)
 * hipcc -o rush_larsen_hip -O3 rush_larsen_hip.cc
 * hipcc -o rush_larsen_hip -O3 "-DCOMPILERID=rocm-4.0.1" rush_larsen_hip.cc
 *
 * Run with no arguments for argument info:
 *   Usage: ./rush_larsen_hip  Iterations  Kernel_GBs_used
 *
 *     Measure launch overhead: ./rush_larsen_hip 100000 .00000001
 *     Measure GPU performance: ./rush_larsen_hip    100 10
 * 
 * 
 * Rush Larsen CUDA/Openmp kernels written by Rob Blake (LLNL) Sept 2016.
 * 
 * Ported to HIP and added inline performance measurements (nvprof not needed)
 * by John Gyllenhaal (gyllenhaal1@llnl.gov) at LLNL 11/10/20.
 *
 * V1.1 03/22/21 command line args, perf diffs, checks return codes and answer
 * V1.0 11/10/20 initial release, hard coded inputs, no error checking
 */

#include "hip/hip_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>

/* Allow version to be printed in output */
static const char *version_string="Version 1.1 (3/22/21)";

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
 * and writes to stderr.  Also flushes stdout, so messages stay
 * in reasonable order.  Now also prints diff time from last printf.
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
    fprintf (stderr, "%7.3f ERROR: %s\n", sec, buf);

    va_end (args);
    exit(1);
}

/*
 * This polynomial is a fit to the dynamics of a small part of a cardiac
 * myocyte, specifically the fast sodium m-gate described here:
 * https://www.ncbi.nlm.nih.gov/pubmed/16565318
 *
 * Does exactly the same work on every cell.   Can scale from one cell
 * to filling entire GPU memory.   Does use cell's value as input
 * to calculations.
 */
__global__ void doRushLarsen(double* m_gate, const int nCells, const double* Vm) {
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if (ii > nCells) { return; }
    
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
}

/* Sets up and calls the doRushLarsen HIP GPU kernel above 'iterations' times,
 * on gpu gpu_id,  allocating GPU arrays one to consume GPU_mem_used GBs
 * of GPU memory.   The desc is expected to be used in multi-compiler 
 * version of this benchmark later.
 *
 * See description of algorithm above in hip kernel.
 */
void rush_larsen_hip(int gpu_id, long iterations, double GPU_mem_used, const char *desc)
{
    hipError_t err;
    double kernel_starttime,kernel_endtime;
    double transfer_starttime,transfer_endtime;
    long nCells;
    long status_point;

    /* Pick GPU to use */
    err = hipSetDevice(gpu_id);
    if (err != hipSuccess)
    {
        punt ("Error HIP%s: GPU %i failed hipSetDevice: %s", desc, gpu_id, hipGetErrorString(err));
    } 
  
    /* Make sure GPU is not totally broken before doing test */
    double* dummyfirst;
    err=hipMalloc(&dummyfirst, sizeof(double)*4);
    if (err != hipSuccess)
    {
        punt ("Error HIP%s: GPU %i failed first hipMalloc: %s", desc, gpu_id, hipGetErrorString(err));
    }
    err=hipFree(dummyfirst);
    if (err != hipSuccess)
    {
        punt ("Error HIP%s: GPU %i failed first hipFree: %s", desc, gpu_id,  hipGetErrorString(err));
    }

    /* For print niceness, make .00000001 lower bound on GB memory */
    if (GPU_mem_used < .00000001)
	GPU_mem_used = .00000001;

    /* Calculate nCells from target memory target */
    nCells = (long) ((GPU_mem_used * 1024.0 * 1024.0 * 1024.0) / (sizeof(double) * 2));

    /* Must have at least 1 cell */
    if (nCells < 1)
        nCells = 1;

    /* Must have at least 1 iteration */
    if (iterations < 1)
        iterations=1;

    /* Give status every 10% of iterations */
    status_point=iterations/10;
    /* Must be at least 1 to make mod work*/
    if (status_point < 1)
        status_point = 1;
    
    printf_timestamp("START Rush Larsen %ld iters %ld cells %.8f GBs HIP%s\n", iterations, nCells, GPU_mem_used, desc); 
    printf_timestamp("%s\n", version_string); 
    printf_timestamp("Allocating and initializing CPU arrays\n");
    double* m_gate = (double*)calloc(nCells,sizeof(double));
    if (m_gate == NULL)
    {
        punt ("GPU %i failed calloc m_gate", gpu_id);
    }
 
    double* Vm = (double*)calloc(nCells,sizeof(double));
    if (Vm == NULL)
    {
        punt ("GPU %i failed calloc Vm", gpu_id);
    }
    
    printf_timestamp("Allocating GPU arrays\n"); 
    double* c_m_gate;
    err=hipMalloc(&c_m_gate, sizeof(double)*nCells);
    if (err != hipSuccess)
    {
        punt ("GPU %i failed hipMalloc c_m_gate: %s", gpu_id, hipGetErrorString(err));
    }
    double* c_Vm;
    err=hipMalloc(&c_Vm, sizeof(double)*nCells);
    if (err != hipSuccess)
    {
        punt ("GPU %i failed hipMalloc c_Vm: %s", gpu_id, hipGetErrorString(err));
    }

    printf_timestamp("Starting hipMemcpy CPU arrays to GPU arrays\n"); 

    transfer_starttime=secs_elapsed();
    err=hipMemcpy(c_m_gate, m_gate, sizeof(double)*nCells, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        punt ("GPU %i failed hipMemcpy c_m_gate: %s", gpu_id, hipGetErrorString(err));
    }

    err=hipMemcpy(c_Vm, Vm, sizeof(double)*nCells, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        punt ("GPU %i failed hipMemcpy c_Vm: %s", gpu_id, hipGetErrorString(err));
    }
    transfer_endtime=secs_elapsed();
    printf_timestamp("Finished hipMemcpy CPU arrays to GPU arrays\n"); 
    
    dim3 gridSize, blockSize;
    blockSize.x=512; blockSize.y=1; blockSize.z=1;
    gridSize.x = (nCells + blockSize.x-1) / blockSize.x; gridSize.y=1; gridSize.z=1;
    for (int itime=0; itime<=iterations; itime++) {
        /* Print warmup message for 0th iteration */
	if (itime == 0)
	    printf_timestamp("Launching warmup iteration (not included in timings)\n");
	
        /* Print status every 10% of iterations */
	else if (((itime-1) % status_point) == 0)
	    printf_timestamp("Starting iteration %6li\n", itime);

        /* Start timer after warm-up iteration 0 */
	if (itime==1)
	    kernel_starttime=secs_elapsed();
        
        /* Launch kernel and wait for results */
	hipLaunchKernelGGL(doRushLarsen, dim3(gridSize), dim3(blockSize), 0, 0, c_m_gate, nCells, c_Vm);
	err = hipDeviceSynchronize();
        if (err != hipSuccess)
        {
            punt ("GPU %i failed hipDeviceSynchronize iteration %li: %s", gpu_id, iterations, hipGetErrorString(err));
        }
    }

    /* Get time after all iterations */
    kernel_endtime=secs_elapsed();

    /* Print time stats */
    printf_timestamp("STATS Rush Larsen %li iters %.4lf sec %.2lf us/iter %.3lf sec datatrans HIP%s\n",  iterations, kernel_endtime-kernel_starttime, (double)(kernel_endtime-kernel_starttime)*1000000.0/(double) iterations, transfer_endtime-transfer_starttime, desc);

    /* Get back data for just m_gate[0] (for now) so can sanity check GPU calculations */
    err=hipMemcpy(m_gate, c_m_gate, sizeof(double), hipMemcpyDeviceToHost);
    if (err != hipSuccess)
    {
        punt ("GPU %i failed hipMemcpy back to m_gate: %s", gpu_id, hipGetErrorString(err));
    }

    /* Sanity check that GPU not giving back garbage
     * Found m_gate[0] to be ~.0.506796353074569 after 1 iteration (really 2 with warmup)
     * and converges to 0.996321172062538 after 100 iterations.  Make sure in that bounds
     * for now.  With a little slop (~.000001) for now (not sure rounding error expected)
     */
    if (m_gate[0] < 0.506796) 
    {
	printf_timestamp("FAILED GPU Data sanity check m_gate[0]=%.15lf < 0.506796 (0.506796353074569 min expected value) HIP%s\n", m_gate[0], desc);
    }
    else if (m_gate[0] > 0.996322)
    {
	printf_timestamp("FAILED GPU Data sanity check m_gate[0]=%.15lf > 0.996322 (0.996321172062538 max expected value) HIP%s\n", m_gate[0], desc);
    }
    else
    {
	printf_timestamp("PASSED GPU Data sanity check m_gate[0]=%.15lf HIP%s\n", m_gate[0], desc);
    }


    /* Free GPU memory */
    err=hipFree(c_Vm);
    if (err != hipSuccess)
    {
        punt ("Error HIP%s: GPU %i failed hipFree c_Vm: %s", desc, gpu_id,  hipGetErrorString(err));
    }

    err=hipFree(c_m_gate);
    if (err != hipSuccess)
    {
        punt ("Error HIP%s: GPU %i failed hipFree c_m_gate: %s", desc, gpu_id,  hipGetErrorString(err));
    }

    /* Free CPU Memory */
    free(Vm);
    free(m_gate);


    printf_timestamp("DONE Freed CPU and GPU memory HIP%s\n", desc);
}

int main(int argc, char* argv[]) 
{
    long max_iterations=1;
    double GPU_mem_used=0.0;
    char tag[100];

    if (argc != 3)
    {
        printf ("Usage: %s  Iterations  Kernel_GBs_used\n", argv[0]);
        printf ("\n");
        printf ("Measure launch overhead: %s 100000 .00000001\n", argv[0]);
        printf ("Measure GPU performance: %s    100 10\n", argv[0]);
        printf ("\n");
        printf ("%s\n", version_string);
        printf ("\n");
        printf ("Questions? Contact John Gyllenhaal (gyllenhaal1@llnl.gov)\n");
        exit (1);
    }

    /* Get iteration count and target GPU memory used arguments */
    max_iterations = atol(argv[1]);
    GPU_mem_used=atof(argv[2]);

    /* The CODETAG defined indirectly by -DCOMPILERID on compile line */
#ifdef COMPILERID
#define rushxstr(s) rushstr(s)
#define rushstr(s) #s
      snprintf(tag, sizeof(tag), " [%s]", rushxstr(COMPILERID));
#else
      strcpy(tag, "");
#endif
   
    rush_larsen_hip(0, max_iterations, GPU_mem_used, tag);

    return(0);
}

