/*
  Copyright (c) 2019-21, Lawrence Livermore National Security, LLC. and other
  Goulash project contributors LLNL-CODE-795383, All rights reserved.
  For details about use and distribution, please read LICENSE and NOTICE from
  the Goulash project repository: http://github.com/llnl/goulash
  SPDX-License-Identifier: BSD-3-Clause
*/

/* Designed to allow direct performance comparisons between 
 * naively written HIP/CUDA and OpenMP GPU offloading schemes
 * in a variety of coding styles and languages of a 
 * parameterized embarrassingly parallel Rush Larsen kernel. 
 * Also allows testing build systems (including SPACK) handling
 * of complicated build situations that LLNL cares about.
 *
 * Designed to create several single file test variants 
 * where no -D options required to select the variant
 * and no include files are needed.
 * Almost all code in this file is identical between 
 * variants (this is intentional).
 * MPI support is ifdefed out for non-MPI variants.
 * 
 * The key computational kernel can be located by
 * searching for: KERNEL
 * 
 * Designed to create trivial variants of the same
 * test to be compiled with different compilers for
 * the interoperability tests.   This is why most
 * functions are static and the main kernel test
 * is called RUSH_LARSEN_VARIANT (preprocessor
 * defines this and VARIANT_DESC based on 
 * variant configuration).
 *
 * The naming convention of the variant copies is
 * intended to indicate variant abilities:
 * _cpu_serial      - single threaded, no OpenMP, on CPU
 * _cpu_omp         - use OpenMP to spawn threads on CPU
 * _gpu_omp         - uses OpenMP to offload to GPU 
 * _gpu_hip         - uses HIP to offload to AMD or Nvidia GPU
 * _gpu_lambda_hip  - RAJA-like lambda HIP variant
 * _gpu_cuda        - uses CUDA to offload to Nvidia GPU
 * _gpu_lambda_cuda - RAJA-like lambda CUDA variant
 * *_mpi            - uses and exercises MPI e.g. _gpu_omp_mpi 
 * *_fort           - Fortran version e.g. _gpu_omp_mpi_fort
 *
 * For the interop tests, there is an additional suffix
 * to indicate different copies of the same configuration
 * that are intended to be compiled by different compilers:
 * _compiler1  - E.g., rush_larsen_gpu_omp_compiler1.cc
 *
 * VARIANT_DESC set by preprocessor directives to
 * the configuration of this file.
 *
 * Recommended that a -DCOMPILERID be set to the compiler used to compile each file:
 *
 * /opt/rocm-4.0.1/llvm/bin/clang++  -o rush_larsen_cpu_omp -O3 -g "-DCOMPILERID=rocm-4.0.1" -fopenmp rush_larsen_cpu_omp.cc
 * 
 * Run with no arguments for suggested arguments, for example:
 *   Usage: ./rush_larsen_cpu_omp  Iterations  Kernel_GBs_used
 *
 *     Measure serial launch overhead:   env OMP_NUM_THREADS=1 ./rush_larsen_cpu_omp 100000 .00000001
 *     Measure thread launch overhead:   ./rush_larsen_cpu_omp 100000 .00000001
 *     Measure kernel performance: ./rush_larsen_cpu_omp    100 10
 * 
 * 
 * The Goulash project conceived of and designed by David Richards, 
 * Tom Scogland, and John Gyllenhaal at LLNL Oct 2019.
 * Please contact John Gyllenhaal (gyllenhaal1@llnl.gov) with questions.
 *
 * Rush Larsen core CUDA/OpenMP kernels written by Rob Blake (LLNL) Sept 2016.
 * The goulash Rush Larsen tests add benchmarking infrastructure 
 * around this incredibly useful compact GPU test kernel.   Thank you Rob!
 * 
 * Inline performance measurements added (nvprof not needed)
 * by John Gyllenhaal at LLNL 11/10/20.
 *
 * Command line argument handling, performance difference printing in
 * form easily found with grep, OpenMP thread mapping and initial data
 * sanity checks on just the first array element calculated by kernel
 * by John Gyllenhaal at LLNL 03/22/21
 * 
 * Pulled code from print_openmp_mapping.c by John Gyllenhaal at
 * LLNL written June 2020 which was based on mpibind tests
 * (https://github.com/LLNL/mpibind) by Edgar Leon at LLNL
 *
 * RAJA-perf-suite-like (https://github.com/LLNL/RAJAPerf)
 * C++ lambda versions created by Jason Burmark at LLNL 06/16/21
 * 
 * C-like Fortran ports (by hand) of C++ version of 
 * rush larsen variants by John Gyllenhaal at LLNL 06/28/21.
 * 
 * MPI stat aggregation and MPI exercisers written
 * by John Gyllenhaal at LLNL based on previous user issue reproducers.
 * Pulled into rush larsen tests on 07/02/21 by John Gyllenhaal at LLNL.
 *
 * Enhanced data checks of all kernel generated array data, including 
 * across MPI ranks by John Gyllenhaal at LLNL 07/03/21
 *
 * Interop versions create by John Gyllenhaal at LLNL 07/14/21
 * to test mixing all the Rush Larsen tests with multiple GPU compilers
 * all in one final executable.
 * 
 * Initial test generator from template files, including Makefiles
 * created by John Gyllenhaal at LLNL 07/21/21 for V2.0RC1
 *
 * V2.0RC1 07/21/21 Added MPI support, interop version, enhanced data checks.
 * V1.2 06/28/21 Fortran and C++ lambda versions added, consistent use of long type
 * V1.1 03/22/21 command line args, perf diffs, maps, checks return codes and answer
 * V1.0 11/10/20 initial release, hard coded inputs, no error checking
 */

/* Allow version to be printed in output */
#define VERSION_STRING "Version 2.0 RC1 (7/21/21)"

/*
 * Set -DGOULASH_UNLOCK to be able to overwrite hard coded variant settings
 * on compile line.
 * 
 * By default, each variant of this file is locked to a particular configuration, 
 * ignoring compile line options.   This is the recommended usage mode!
 *
 * These additional locked variant suboptions can be used to
 * compile out some optional functionality of the tests:
 *
 *   NO_THREAD_MAP     - Do not print threadmap for TARGET_CPU_OMP
 *   NO_MPI_EXERCISERS - Do not exercise MPI with MPI variants
 *   NO_STATIC         - Do not make support calls static scope
 *   NO_MAIN           - Do not use main driver, used for interop tests
 *
 * Unlocking designed to allow regression testing on all supported variants
 * from one source file before copying and hard coding these settings
 * for each variant when pushing out a new Goulash version.
 */

/* Unless unlocked, define variant for this file. */
#ifndef GOULASH_UNLOCK

/* MUST BE EXACTLY TWO SPACES BETWEEN #undef AND VARIABLE */
/* THIS IS REQUIRED FOR THE TEMPLATE INSTANTIATION */

/* Exactly one TARGET must be defined to determine variant used */
#undef  TARGET_CPU_SERIAL
#undef  TARGET_CPU_OMP
#undef  TARGET_GPU_OMP
#undef  TARGET_GPU_HIP
#define TARGET_GPU_LAMBDA_HIP
#undef  TARGET_GPU_CUDA
#undef  TARGET_GPU_LAMBDA_CUDA

/* Any target may use MPI to aggregate results and exercise MPI */
#define USE_MPI

/* Used by interop tests to append id to routine's name (e.g., _compiler1) */
#define VARIANT_ID _compiler1

/* Additional variant suboptions
 * NOTE: Cannot have comments on same lines as #undef due to generation logic!
 */
/* Do not print threadmap when have TARGET_CPU_OMP */
#undef  NO_THREAD_MAP

/* Do not exercise MPI with MPI variants */
#undef  NO_MPI_EXERCISERS

/* Do not make support calls static scope */
#undef  NO_STATIC

/* Do not use main driver, used for interop tests */
#define NO_MAIN

#endif /* !GOULASH_UNLOCK */

/* nvcc does not like these preprocessor directives */
#ifdef __CUDA__
/*
 * Make sure exactly one TARGET is defined 
 * Generate preprocessing error otherwise.
 */
#if !((defined(TARGET_CPU_SERIAL) + defined(TARGET_CPU_OMP) + defined(TARGET_GPU_OMP) + defined(TARGET_GPU_HIP) + defined(TARGET_GPU_LAMBDA_HIP) + defined(TARGET_GPU_CUDA) + defined(TARGET_GPU_CUDA)) == 1)
/* Do not have exactly one target
 * Indicate if zero targets or multiple targets is the problem 
 */
#if ((defined(TARGET_CPU_SERIAL) + defined(TARGET_CPU_OMP) + defined(TARGET_GPU_OMP) + defined(TARGET_GPU_HIP) + defined(TARGET_GPU_LAMBDA_HIP) + defined(TARGET_GPU_CUDA) + defined(TARGET_GPU_CUDA)) == 0)
#error Zero TARGETs defined.   Exactly one TARGET must be defined to compile rush_larsen from Goulash
#else
#error Multiple TARGETs defined.   Exactly one TARGET must be defined to compile rush_larsen from Goulash
#endif

/* With some preprocessors these tests will show what is multiply defined */
#ifdef TARGET_CPU_SERIAL
#error Multiple targets specified: TARGET_CPU_SERIAL
#endif
#ifdef TARGET_CPU_OMP
#error Multiple targets specified: TARGET_CPU_OMP
#endif
#ifdef TARGET_GPU_OMP
#error Multiple targets specified: TARGET_GPU_OMP
#endif
#ifdef TARGET_GPU_HIP
#error Multiple targets specified: TARGET_GPU_HIP
#endif
#ifdef TARGET_GPU_LAMBDA_HIP
#error Multiple targets specified: TARGET_GPU_LAMBDA_HIP
#endif
#ifdef TARGET_GPU_CUDA
#error Multiple targets specified: TARGET_GPU_CUDA
#endif
#ifdef TARGET_GPU_LAMBDA_CUDA
#error Multiple targets specified: TARGET_GPU_LAMBDA_CUDA
#endif
#endif
#endif


/* If NO_STATIC defined, make all support routines non-static (visible) */
#ifdef NO_STATIC
#define STATIC
#else
#define STATIC static
#endif

/* Preprocessor macro rushglue(x,y) glues two defined values together */
#define rushglue2(x,y) x##y
#define rushglue(x,y) rushglue2(x,y) 

/* Preprocessor macro rushxstr(s) converts value to string */
#define rushxstr2(s) #s
#define rushxstr(s) rushxstr2(s)

/* Preprocessing macro to check return code from HIP calls using local scope err */
#define HIPCHECK(x) {hipError_t err=x; if (err != hipSuccess) punt ("ERROR HIP failed: " rushxstr(x) ": %s", hipGetErrorString(err));}

/* Preprocessing macro to check return code from CUDA calls using local scope err */
#define CUDACHECK(x) {cudaError_t err=x; if (err != cudaSuccess) punt ("ERROR CUDA failed: " rushxstr(x) ": %s", cudaGetErrorString(err));}

/* For interop version, can #define VARIANT_ID 1, etc. to make different named kernels.
 * If not defined, make empty string 
 */
#ifndef VARIANT_ID
#define VARIANT_ID
#endif

/* Create unique VARIANT_TAG based on file #def and #undef
 * settings that is used to create rush_larsen function
 * call name and to annotate key lines of output
 */
#undef TARGET_TAG
#if   defined(TARGET_CPU_SERIAL)
#define TARGET_TAG cpu_serial
#elif defined(TARGET_CPU_OMP)
#define TARGET_TAG cpu_omp
#elif defined(TARGET_GPU_OMP)
#define TARGET_TAG gpu_omp
#elif defined(TARGET_GPU_HIP)
#define TARGET_TAG gpu_hip
#elif defined(TARGET_GPU_LAMBDA_HIP)
#define TARGET_TAG gpu_lambda_hip
#elif defined(TARGET_GPU_CUDA)
#define TARGET_TAG gpu_cuda
#elif defined(TARGET_GPU_LAMBDA_CUDA)
#define TARGET_TAG gpu_lambda_cuda
#else
#error No matching TARGET_TAG found
#endif

/* Append _mpi to indicate using MPI */
#undef RUSH_MPI_TAG
#if defined(USE_MPI)
#define RUSH_MPI_TAG rushglue(TARGET_TAG,_mpi)
#else
#define RUSH_MPI_TAG TARGET_TAG
#endif

/* Append VARIANT_ID (may be empty) to get VARIANT_TAG */
#undef VARIANT_TAG
#define VARIANT_TAG rushglue(RUSH_MPI_TAG,VARIANT_ID)

/* Create name of rush larsen kernel routine from all variant info and the VARIANT ID */
#define RUSH_LARSEN_VARIANT rushglue(rush_larsen_,VARIANT_TAG)

/* Generate VARIANT_DESC string that annotates the end of key output
 * lines spread across this whole file.  Uses C trick that
 * "omp" " [" "g++" "]" 
 * is equivalent to
 * "omp [g++]"
 * Since I could not figure out how to create one big string
 * with the preprocessor.
 */
#ifdef COMPILERID
#define VARIANT_DESC rushxstr(VARIANT_TAG) " ["  rushxstr(COMPILERID) "]"
#else
#define VARIANT_DESC rushxstr(VARIANT_TAG)
#endif

/* Only include OpenMP for variants that need it */
#if defined(TARGET_CPU_OMP) || defined(TARGET_GPU_OMP)
#include <omp.h>
#endif

/* Only include HIP for variants that need it */
#if defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP)  
#include "hip/hip_runtime.h"
#endif
/* Only include CUDA for variants that need it */
#if defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA)  
#include "cuda_runtime.h"
#endif
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sched.h>

/* Only include mpi for variants that need it */
#ifdef USE_MPI
#include "mpi.h"
#endif

#if 0
    ; /* Fix emacs indent program logic */
#endif

/* Get raw time in seconds as double (a large number).
 * Returns -1.0 on unexpected error.
 */
STATIC double get_raw_secs( void )
{
    struct timeval ts;
    int status;
    double raw_time;
        
    /* Get wall-clock time */
    /* status = getclock( CLOCK_REALTIME, &ts ); */
    status = gettimeofday( &ts, NULL );
        
    /* Return -1.0 on error */
    if( status != 0 ) return -1.0;
        
    /* Convert structure to double (in seconds ) (a large number) */
    raw_time = (double)ts.tv_sec + (double)ts.tv_usec * 1e-6;

    return (raw_time);
}
        
/* Returns base time.  If new_time >= 0, 
 * sets base_time to new_time before returning.
 * Using this as access method to static variable
 * in a way I can trivially emulate in fortran.
 *
 * Note: Lock shouldn't be needed, since even if multiple
 *       threads initialize this, it will be to basically
 *       the same value.
 */
STATIC double get_base_time(double new_time)
{
    static double base_time = -1.0;

    /* If passed value >= 0, use as new base_time */ 
    if (new_time >= 0.0)
        base_time = new_time;

    return(base_time);
}

/* Returns time in seconds (double) since the first call to secs_elapsed
 * (i.e., the first call returns 0.0).
 */
STATIC double secs_elapsed( void )
{
    double new_time;
    double base_time;
        
    /* Get current raw time (a big number) */
    new_time = get_raw_secs();
        
    /* Get the offset since first time called (pass -1 to query)*/
    base_time = get_base_time(-1.0);
    
    /* If base time not set (negative), set to current time (pass in positive secs)*/
    if (base_time < 0.0)
        base_time = get_base_time(new_time);
     
    /* Returned offset from first time called */
    return (new_time - base_time);
}

/* Works like vfprintf, except prefixes wall-clock time (using secs_elapsed)
 * and the difference since last vfprintf.
 * Also flushes out after printing so messages appear immediately
 * Used to implement printf_timestamp, punt, etc.
 */
STATIC double last_printf_timestamp=0.0;
STATIC void vfprintf_timestamp (FILE*out, const char * fmt, va_list args)
{
    char buf[4096];
    int rank = -1;  /* Don't print rank for serial runs */
#ifdef USE_MPI
    /* Get actual MPI rank if using MPI */
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif

    /* Get wall-clock time since first call to secs_elapsed */
    double sec = secs_elapsed();
    double diff = sec - last_printf_timestamp;
    last_printf_timestamp=sec;

    /* Print out passed message to big buffer*/
    vsnprintf (buf, sizeof(buf), fmt, args);

    /* No MPI case */
    if (rank < 0)
    {
        /* Print out timestamp and diff seconds with buffer*/
        fprintf (out, "%7.3f (%05.3fs): %s", sec, diff, buf);
    }
    /* MPI case, add rank */
    else
    {
        /* Print out timestamp and diff seconds and MPI rank with buffer*/
        fprintf (out, "%3i: %7.3f (%05.3fs): %s", rank, sec, diff, buf);
    }

    /* Flush out, so message appears immediately */
    fflush (out);
}

/* Prints to stdout for all MPI ranks with timestamps and time diffs */
STATIC void printf_timestamp (const char * fmt, ...)
{
    va_list args;
    va_start (args, fmt);

    /* Use helper routine to actually do print and flush */
    vfprintf_timestamp (stdout,fmt,args);

    va_end (args);
}

/* Prints to stdout for only Rank 0 with timestamps and time diffs.
 * For all other MPI ranks, the message is thrown away.
 */
STATIC void rank0_printf_timestamp (const char * fmt, ...)
{
    int rank = 0;  /* Non-mpi runs always get printed out */
    va_list args;
    va_start (args, fmt);

#ifdef USE_MPI
    /* Get real rank if actually using MPI*/
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif

    /* Only print if rank 0 (or non-MPI program) */
    if (rank == 0)
    {
        /* Use helper routine to actually do print and flush */
        vfprintf_timestamp (stdout,fmt,args);
    }

    va_end (args);
}

/* Prints to stderr (flushes stdout first) with timestamp and exits */
STATIC void punt (const char * fmt, ...)
{
    va_list args;
    va_start (args, fmt);

    /* Flush stdout, so pending message appears before punt message */
    fflush (stdout);

    /* Use helper routine to actually do print and flush */
    vfprintf_timestamp (stderr,fmt,args);

    va_end (args);

    /* Abort the program */
    exit(1);
}

/* The maximum number of threads supported */
#define MAX_SIZE 1024

#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP)
/* Spawns OpenMP threads and prints cpu mappings.
 * Send rank -1 if not in MPI program (or don't want rank printed).
 */
STATIC int print_openmp_mapping (const char *location, int rank)
{
    int map_array[MAX_SIZE+10];
    char tag[1050]="";
    char env_str[50000]="";
    char *omp_proc_bind_str=NULL, *omp_places_str=NULL, *omp_num_threads_str=NULL;
    char map_buf[10000];
    int num_threads=-1;
    char host[1024]="unknown";
    int i;

    /* Get OMP env variables that could affect binding */
    omp_proc_bind_str = getenv("OMP_PROC_BIND");
    omp_places_str = getenv("OMP_PLACES");
    omp_num_threads_str = getenv("OMP_NUM_THREADS");

    /* Add environment variables to env_str if set */
    if (omp_num_threads_str != NULL)
        sprintf (env_str+strlen(env_str), " OMP_NUM_THREADS=%s",  omp_num_threads_str);

    if (omp_proc_bind_str != NULL)
        sprintf (env_str+strlen(env_str), " OMP_PROC_BIND=%s",  omp_proc_bind_str);

    if (omp_places_str != NULL)
        sprintf (env_str+strlen(env_str), " OMP_PLACES=%s",  omp_places_str);

    gethostname(host, sizeof(host));


    if (rank >= 0)
    {
        sprintf (tag, "Rank %2i %s", rank, host);
    }
    else
    {
        sprintf (tag, "%s", host);
    }

    /* Initialize cpu_id array to -1 */
    for (i=0; i < MAX_SIZE; i++)
        map_array[i] = -1;
    /* Mark 1 past array as endpoint to simplify logic */
    map_array[MAX_SIZE] = -1;

/* Put enough work that should hit all cpu ids available to OpenMP */
#pragma omp parallel for
    for (i=0; i < 1000000; i++)
    {
        /* Mark what cpu_id used by iteration i */
        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();

        if ((cpu >= 0) && (cpu < MAX_SIZE))
        {
            /* Racy update but ok, just need it not to be -1 */
            if (map_array[cpu] < i)
                map_array[cpu] = i;
        }
        else
        {
            printf ("Unexpected tid %i cpu %i\n", tid, cpu);
        }
    }

    /* Create string with concise listing of cpus used */
    num_threads=0;
    sprintf (map_buf, "Map ");
    for (i=0; i < MAX_SIZE; i++)
    {
        int start=-1, end=-1;
        /* Create string of cpu ids used by OpenMP threads */
        if (map_array[i] != -1)
        {
            /* Add comma if not first entry */
            if (num_threads > 0)
                sprintf (map_buf+strlen(map_buf), ",");

            start=i;
            num_threads++;
            if (map_array[i+1] != -1)
            {
                /* Count continuous thread numbers */
                while (map_array[i+1] != -1)
                {
                    num_threads++;
                    i++;
                }
                end=i;
                sprintf (map_buf+strlen(map_buf), "%i-%i",  start, end);
            }
            else
            {
                sprintf (map_buf+strlen(map_buf), "%i",  i);
            }
        }
    }

    /* print out one line per process */
    printf_timestamp ("%s %s Threads %i %s%s\n", location, map_buf, num_threads, tag, env_str);
    return (num_threads);
}

/* For non-MPI programs, turns into direct call to print_openmp_mapping 
 * 
 * For MPI programs, syncs ranks, and print thread stats across ranks,
 * in addition to calling print_openmp_mapping
 */
STATIC void print_thread_stats (const char *location)
{
#ifdef USE_MPI
    double numthreads, maxnumthreads, minnumthreads, sumnumthreads, avgnumthreads;
    int numtasks, ret_code;

    /* Get number of tasks */ 
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    /* Wait for all MPI ranks to complete */
    MPI_Barrier (MPI_COMM_WORLD);

    rank0_printf_timestamp("Printing %s mapping (all tasks)\n", location);

    /* Synchronize printf timestamp diffs across MPI ranks */
    last_printf_timestamp= secs_elapsed();
#endif

    /* Print OpenMP thread mapping
     * Turn off Rank printing in call since using
     * printf_timestamp's rank printing already
     * Only capture intnumthreads in MPI mode to make -Wall clean
     */
#ifdef USE_MPI
    int intnumthreads = 
#endif
        print_openmp_mapping (location, -1);

#ifdef USE_MPI
    /* Make numthreads a double to make easier to work with */
    numthreads = intnumthreads;

    ret_code = MPI_Allreduce (&numthreads, &maxnumthreads, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce max numthreads returned %i\n", ret_code);

    ret_code = MPI_Allreduce (&numthreads, &minnumthreads, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce min numthreads returned %i\n", ret_code);

    ret_code = MPI_Allreduce (&numthreads, &sumnumthreads, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce sum numthreads returned %i\n", ret_code);
    avgnumthreads = sumnumthreads / (double) numtasks;

    /* Don't print thread stats for no OpenMP case */
    rank0_printf_timestamp("THREADSTATS %s %i tasks  min %.0f  avg %.1f  max %.0f  maxdiff %.2f%%  %s\n",  
                           location, numtasks, minnumthreads, avgnumthreads, maxnumthreads, (maxnumthreads-minnumthreads)*100.0/minnumthreads, VARIANT_DESC);
    /* Sync all ranks on exit */
    MPI_Barrier (MPI_COMM_WORLD);
#endif
}
#endif /* defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP) */

/* Print kernel runtime stats and aggregate across MPI processes if necessary.
 * Prints one liner if not using MPI
 */
STATIC void print_runtime_stats(long iterations, double kernel_mem_used, double kernel_runtime, double transfer_runtime)
{
#ifdef USE_MPI
    double kernel_maxruntime, kernel_minruntime, kernel_sumruntime, kernel_avgruntime, kernel_variation;
    double transfer_maxruntime, transfer_minruntime, transfer_sumruntime, transfer_avgruntime, transfer_variation;
    int  numtasks, ret_code;

    /* Get number of tasks */ 
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    /* Print stats only if more than 1 rank */
    if (numtasks > 1)
    {
        rank0_printf_timestamp("Collecting and aggregating kernel runtimes across MPI ranks\n");
    }

    ret_code = MPI_Allreduce (&kernel_runtime, &kernel_maxruntime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce max runtime returned %i\n", ret_code);

    ret_code = MPI_Allreduce (&kernel_runtime, &kernel_minruntime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce min runtime returned %i\n", ret_code);

    ret_code = MPI_Allreduce (&kernel_runtime, &kernel_sumruntime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce sum runtime returned %i\n", ret_code);
    kernel_avgruntime = kernel_sumruntime / (double) numtasks;
    if (kernel_minruntime > 0.0001)
        kernel_variation = (kernel_maxruntime - kernel_minruntime)*100.0/kernel_minruntime;
    else 
        kernel_variation = (kernel_maxruntime - kernel_minruntime)*100.0/0.0001;

    ret_code = MPI_Allreduce (&transfer_runtime, &transfer_maxruntime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce max runtime returned %i\n", ret_code);

    ret_code = MPI_Allreduce (&transfer_runtime, &transfer_minruntime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce min runtime returned %i\n", ret_code);

    ret_code = MPI_Allreduce (&transfer_runtime, &transfer_sumruntime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce sum runtime returned %i\n", ret_code);
    transfer_avgruntime = transfer_sumruntime / (double) numtasks;
    if (transfer_minruntime > 0.0001)
        transfer_variation = (transfer_maxruntime - transfer_minruntime)*100.0/transfer_minruntime;
    else 
        transfer_variation = (transfer_maxruntime - transfer_minruntime)*100.0/0.0001;


    /* Print time stats with min and max  if more than 1 rank */
    if (numtasks > 1)
    {
        rank0_printf_timestamp("VARIATION kernel %7.2lf%%  datatrans %7.2lf%%  %i tasks  Rush Larsen %li %.8f  %s\n",  
                               kernel_variation, transfer_variation,  
                               numtasks, iterations, kernel_mem_used, VARIANT_DESC);
        rank0_printf_timestamp("MINSTATS   Rush Larsen %li %.8f  %.4lf s  %.2lf us/iter  %.3f s datatrans %s\n",  
                               iterations, kernel_mem_used, kernel_minruntime, (double)(kernel_minruntime)*1000000.0/(double) iterations, transfer_minruntime, VARIANT_DESC);
        rank0_printf_timestamp("AVGSTATS   Rush Larsen %li %.8f  %.4lf s  %.2lf us/iter  %.3f s datatrans %s\n",  
                               iterations, kernel_mem_used, kernel_avgruntime, (double)(kernel_avgruntime)*1000000.0/(double) iterations, transfer_avgruntime, VARIANT_DESC);
        rank0_printf_timestamp("MAXSTATS   Rush Larsen %li %.8f  %.4lf s  %.2lf us/iter  %.3f s datatrans %s\n",  
                               iterations, kernel_mem_used, kernel_maxruntime, (double)(kernel_maxruntime)*1000000.0/(double) iterations, transfer_maxruntime, VARIANT_DESC);
    }
    /* Our apps run in lockstep, so MAX time drives cycle time, so we use max for RUSHSTATS */
    /* That said, we don't do sync across tasks every iterations, so worse in real apps */
    rank0_printf_timestamp("RUSHSTATS  Rush Larsen %li %.8f  %.4lf s  %.2lf us/iter  %.3f s datatrans %s\n",  
                           iterations, kernel_mem_used, kernel_maxruntime, (double)(kernel_maxruntime)*1000000.0/(double) iterations, transfer_maxruntime, VARIANT_DESC);

/* NO MPI CASE - print one line */
#else
    /* Print time stats */
    printf_timestamp("RUSHSTATS  Rush Larsen %li %.8f  %.4lf s  %.2lf us/iter  %.3f s datatrans %s\n",  
                     iterations, kernel_mem_used, kernel_runtime, (double)(kernel_runtime)*1000000.0/(double) iterations, transfer_runtime, VARIANT_DESC);
#endif
}

/* Do sanity and consistency checks on all of m_gate. Including cross-rank if MPI mode 
 * Prints PASS or FAIL based on data check results
 * If bad data found, will print up to 5 lines of debug info per MPI rank.
 * Returns fail count so can be returned to caller.
 */
STATIC long data_check (double *m_gate, long iterations, double kernel_mem_used, long nCells)
{
    /* If in MPI mode, in order to prevent MPI hangs on data fails, 
     * need to do MPI allreduces even if earlier checks fail.   
     * As a bonus, this algorithm allows rank 0 to be able to 
     * print out how many data check failures occurred across all ranks.
     */
    long fail_count = 0;

#ifdef USE_MPI
    double dfail_count, aggregate_fail_count=0; /* Make double so can use MPI to aggregate */
    double checkval, rank0checkval, mincheckval, maxcheckval;
    int rank;
    int ret_code;

    /* Get actual MPI rank if using MPI */
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);

    /* Synchronize printf timestamps across MPI ranks */
    last_printf_timestamp= secs_elapsed();
#endif

    rank0_printf_timestamp("Starting data check for sanity and consistency\n");

    /* Sanity check that calculation not giving garbage
     * Found m_gate[0] to be ~.0.506796353074569 after 1 iteration (really 2 with warmup)
     * and converges to 0.996321172062538 after 100 iterations.  Make sure in that bounds
     * for now.  With a little slop (~.000001) for now (not sure rounding error expected)
     */
    if (m_gate[0] < 0.506796) 
    {
        printf_timestamp("ERROR Data sanity check m_gate[0]=%.15lf < 0.506796 (0.506796353074569 min expected value) %s\n", m_gate[0], VARIANT_DESC);
        fail_count++;
    }

    if (m_gate[0] > 0.996322)
    {
        printf_timestamp("ERROR Data sanity check m_gate[0]=%.15lf > 0.996322 (0.996321172062538 max expected value) %s\n", m_gate[0], VARIANT_DESC);
        fail_count++;

    }

    /* Every array entry should have the same value as m_gate[0], make sure that is true */
    for (long i = 1; i < nCells; i++)
    {
        if (m_gate[i] != m_gate[0])
        {
            fail_count++;
            /* Only print at most 5 warnings per rank */
            if (fail_count < 5)
            {
                printf_timestamp("ERROR Data consistency check m_gate[%i]=%.15lf != m_gate[0]=%.15lf %s\n", 
                                 i, m_gate[i], m_gate[0], VARIANT_DESC);
            }
            if (fail_count == 5)
            {
                printf_timestamp("ERROR Data consistency check REMAINING ERROR MESSAGES SUPPRESSED! %s\n", VARIANT_DESC);
            }

        }
    }

    /* Value looks ok, check all ranks match if using MPI */
#ifdef USE_MPI
 
    /* With MPI, check that every rank gets same value for m_gate[0] */
    /* Every task does its own checking of the rest of the values against m_gate[0] */

    /* Get the kernel result we are checking */
    checkval=m_gate[0];

    /* Everyone should check against rank 0's value */
    if (rank == 0)
        rank0checkval = checkval;
    else
        rank0checkval = -1;

    ret_code =  MPI_Bcast(&rank0checkval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Bcase rank 0's checkval returned %i\n", ret_code);

    /* Everyone should check against rank 0's check value and print message on mismatch */
    if (m_gate[0] != rank0checkval)
    {
        printf_timestamp("ERROR Data consistency check rank %i's m_gate[0]=%.15lf != rank 0's m_gate[0]=%.15lf %s\n", rank, m_gate[0], rank0checkval, VARIANT_DESC);
        fail_count++;
    }
       
    /* Aggregate the fail count across all processes, convert to DOUBLE since MPI cannot sum MPI_LONG_INT*/
    dfail_count = (double)fail_count;
    aggregate_fail_count = -1;
    ret_code = MPI_Allreduce (&dfail_count, &aggregate_fail_count, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce aggregate fail_count returned %i\n", ret_code);

    /* Set all fail counts to aggregate fail count on all ranks (convert back to long int)*/
    fail_count = (double) aggregate_fail_count;

    /* Allow rank 0 to detect if there was data mismatch on a different rank */
    ret_code = MPI_Allreduce (&checkval, &maxcheckval, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce max checkval returned %i\n", ret_code);
   
    ret_code = MPI_Allreduce (&checkval, &mincheckval, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt ("Error: MPI_Allreduce min checkval returned %i\n", ret_code);


    /* If mismatches between ranks, min and max value values will be different */
    if (maxcheckval != mincheckval)
    {
        rank0_printf_timestamp("ERROR Data consistency check DETECTED MISMATCHES BETWEEN RANKS rank 0's m_gate[0]=%.15lf %s\n",  rank0checkval, VARIANT_DESC);
    }
#endif

    /* Print out summary PASSED or FAILED count from rank 0 only*/
    if (fail_count == 0)
    {
        rank0_printf_timestamp("PASSED Data check %ld %.8f  m_gate[0]=%.15lf %s\n", iterations, kernel_mem_used, m_gate[0], VARIANT_DESC);
    }
    else
    {
        rank0_printf_timestamp("FAILED Data check %ld %.8f  with %li DATA CHECK ERRORS m_gate[0]=%.15lf %s\n", 
                               iterations, kernel_mem_used, fail_count, m_gate[0], VARIANT_DESC);
    }

    /* Return the number of data_check failures detected (across all ranks, if MPI mode) */
    return (fail_count);
}

#if defined(USE_MPI) && !defined(NO_MPI_EXERCISERS)
/* Do some short MPI stress tests that have exposed issues before (if in MPI mode) */

/* This bcast test tends to break MPI when executables are not readable
 * because this disables Linux's shared memory support (MPI has to detect
 * and work around).
 * Generally hits assertion or logs MPI error message.
 * 
 * (Typical test:  chmod -r ./a.out and run 2 or more tasks per node.) 
 * Recommend 8196, 131072.  Usually just 16432,16432 is sufficient to trigger.
 */
STATIC void do_bcast_noshmmem_test (long start_count, long end_count, MPI_Comm comm)
{
    int  rank, numtasks;
    unsigned char *char_array;
    unsigned int  *int_array;
    unsigned char myval, rank0val;
    double *double_array;
    int err_count=0;
    long i, j;
    double start_time, end_time, run_time;

    /* Min acceptable start_count is 1 */
    if (start_count < 1)
        start_count = 1;

    /* Min acceptable end_count is 'start_count' */
    if (end_count < start_count)
        end_count = start_count;

    /* Time all the tests for rough timing info */
    start_time = secs_elapsed();

    /* Allocate array of end_count+1 size to try different sizes of bcast */
    if ((char_array = (unsigned char *) calloc(end_count+1, sizeof(char))) == NULL)
        punt ("do_bcast_noshmem_test: Out of memory allocating %i chars\n", end_count+1);

    if ((int_array = (unsigned int *) calloc(end_count+1, sizeof(int))) == NULL)
        punt ("do_bcast_noshmem_test: Out of memory allocating %i ints\n", end_count+1);

    if ((double_array = (double *) calloc(end_count+1, sizeof(double))) == NULL)
        punt ("do_bcast_noshmem_test: Out of memory allocating %i doubles\n", end_count+1);

    /* Get size of communicator and rank in it */
    MPI_Comm_size (comm, &numtasks);
    MPI_Comm_rank(comm,&rank);

    /* Rank zero value will be 200, everyone else rank %128 */
    rank0val = 200;
    if (rank == 0)
        myval = rank0val;
    else
        myval = rank % 128;
   

    /* Data size dependent, 16384 reproduces for me */
    for (i=start_count; i <= end_count; i*=2)
    {
        /* Try different data types, 16k chars seems sufficient if multiple types and sizes takes too long */
        rank0_printf_timestamp("MPI_Bcast shmem exerciser with %6i chars, ints, and doubles %i tasks\n", i, numtasks);

        /* Initialize data to different values on every rank */
        for (j=0; j < i; j++)
        {
            char_array[j] = myval;
            int_array[j] = myval;
            double_array[j] = myval;
        }

        /* Test the different size/data type bcasts */
        MPI_Bcast(char_array, i, MPI_CHAR, 0, comm);

        MPI_Bcast(int_array, i, MPI_INT, 0, comm);

        MPI_Bcast(double_array, i, MPI_DOUBLE, 0, comm);

        /* Check that everyone got rank 0's value */
        for (j=0; j < i; j++)
        {
            if(char_array[j] != rank0val)
            {
                printf_timestamp ("ERROR: MPI_Bcast exerciser: char_array[%i] = %i (%i expected)\n",
                                  j, char_array[j], rank0val);
                err_count++;
            }
              
            if(int_array[j] != (unsigned int) rank0val)
            {
                printf_timestamp ("ERROR: MPI_Bcast exerciser: int_array[%i] = %i (%i expected)\n",
                                  j, int_array[j], rank0val);
                err_count++;
            }
              
            if(double_array[j] != (double) rank0val)
            {
                printf_timestamp ("ERROR: MPI_Bcast exerciser: double_array[%i] = %i (%i expected)\n",
                                  j, double_array[j], rank0val);
                err_count++;
            }
              
        }
        if (err_count != 0)
            punt("ERROR: MPI_Bcast exercises failures detected.  Exiting\n");
    }

    /* Free buffer arrays */
    free (char_array);
    free (int_array);
    free (double_array);

    /* Time all the tests for rough timing info */
    end_time = secs_elapsed();
    run_time = end_time - start_time;
    rank0_printf_timestamp("MPI_Bcast exerciser total RUNTIME %.4lf s  %i - %i ints %i tasks\n", 
                           run_time, start_count, end_count,  numtasks);
}

/* Several MPIs have gotten intermittent wrong answers (racy) with the the MPI_Allreduce MAX
 * of array values in this exerciser at scale.   Attempt to scale up to larger arrays at small task count
 */
STATIC void do_allreduce_exerciser (int tries, int start_target_size, int end_target_size, MPI_Comm comm)
{
    int *send_array, *reduced_array;
    int numtasks;
    int rank;
    int attempt;
    int i;
    int ret_code;
    int max1, max1o, max2, max2o, max3, max3o;
    int err_count;
    int ints_per_task;
    int target_size, aim_for_size;
    double start_time, end_time, run_time;

    /* Get size of communicator and rank in it */
    MPI_Comm_size (comm, &numtasks);
    MPI_Comm_rank (comm, &rank);

    /* Want at least one int per task and to run the test at least once */
    if (start_target_size < numtasks)
        start_target_size = numtasks;

    /* Want the test to run at least once, so fix end_target_size if needed */
    if (end_target_size < start_target_size)
        end_target_size = start_target_size;

    /* Time all the tests for rough timing info */
    start_time = secs_elapsed();

    /* Test target_sizes in range, doubling every time */
    for (aim_for_size=start_target_size; aim_for_size <= end_target_size; aim_for_size*=2)
    {
    
        /* Every task gets same sized slice of array, may shrink target_size */
        ints_per_task = aim_for_size/numtasks;

        /* Get as big as we can with same size per task */
        target_size = numtasks * ints_per_task;

        /* Allocate send and reduction array */
        send_array = (int *)malloc (sizeof(int) * (target_size));
        reduced_array = (int *)malloc (sizeof(int) * (target_size));

        if ((send_array == NULL) || (reduced_array == NULL))
            punt("do_allreduce_exerciser: Out of memory allocating arrays\n");

        rank0_printf_timestamp("MPI_Allreduce exerciser with %6i ints %i iterations %i tasks\n", target_size, tries, numtasks);

        /* Initialize everything to negative task number so we can track down who's data we got. */
        /* Use negative rank so works with more than 1000000 tasks */
        for (i=0; i < target_size; i++)
        {
            send_array[i] = -rank;
        }
    
        /* Do reduce multiple times with same arrays, etc. (often race issue) */
        for (attempt = 0; attempt < tries; attempt++)
        {
            /* Initial destination array to -1 */
            for (i=0; i < target_size; i++)
                reduced_array[i] = -1;

            /* Set send_array at range assigned to each index to rank + 1000000.
             * At the end of the MAX reduction, every index should be that index + 1000000.
             * If not, it will tell us which data we actually got.
             */
            for (i=rank*ints_per_task; i < (rank+1)*ints_per_task; i++)
            {
                send_array[i] = rank + 1000000;
            }

            /* Create similar MPI noise as original code, do one small allreduce before */
            max1 = rank;
            ret_code = MPI_Allreduce (&max1, &max1o, 1, MPI_INT, MPI_MAX,  comm);
            if (ret_code != MPI_SUCCESS)
                punt("Error: MPI_Allreduce max1 returned %i\n", ret_code);

            /* Do reduce with MAX, so should have every rank's initialization
             * just in their range.
             */
            ret_code = MPI_Allreduce (send_array, reduced_array,
                                      target_size, MPI_INT,
                                      MPI_MAX,  comm);
            if (ret_code != MPI_SUCCESS)
                punt("Error: MPI_Allreduce send_array returned %i\n", ret_code);

            /* Create similar MPI noise as original code, do two small allreduces after */
            max2 = 1;
            ret_code = MPI_Allreduce (&max2, &max2o, 1, MPI_INT, MPI_MAX,  comm);
            if (ret_code != MPI_SUCCESS)
                punt("Error: MPI_Allreduce max2 returned %i\n", ret_code);

            max3 = rank/2;
            ret_code = MPI_Allreduce (&max3, &max3o, 1, MPI_INT, MPI_MAX,  comm);
            if (ret_code != MPI_SUCCESS)
                punt ("Error: MPI_Allreduce max3 returned %i\n", ret_code);

            /* Expect index range to match value if reduction done properly */
            err_count = 0;
            for (i=0; i < target_size; i++)
            {
                /* Each rank gets a range of value here */
                int expected_value = (i/ints_per_task) + 1000000;

                if (reduced_array[i] != expected_value)
                {
                    /* Only print at most 5 warnings per rank */
                    if (err_count < 5)
                    {
                        printf_timestamp ("ERROR: MPI_Allreduce exerciser: reduced_array[%i] = %i (%i expected)\n",
                                          i, reduced_array[i], expected_value);
                    }
                    if (err_count == 5)
                    {
                        printf_timestamp ("ERROR: MPI_Allreduce exerciser REMAINING ERROR MESSAGES SUPPRESSED!\n");
                    }
                    err_count++;
                }
            }
            if (err_count != 0)
                punt("ERROR: MPI_Allreduce exercises %i failures detected.  Exiting\n", err_count);
        }
        /* Free memory */
        free (send_array);
        free (reduced_array);
    }
    /* Time all the tests for rough timing info */
    end_time = secs_elapsed();
    run_time = end_time - start_time;
    rank0_printf_timestamp("MPI_Allreduce exerciser total RUNTIME %.4lf s  %i - %i ints %i iterations %i tasks\n", 
                           run_time, start_target_size, end_target_size, tries, numtasks);
}
 

/* Main stress test driver.  Individual stress tests above */
STATIC void mpi_exerciser_driver(const char *location)
{
    int numtasks;
    MPI_Comm subcomm;
    double start_time, end_time, run_time;

    /* Time all the tests for rough timing info */
    start_time = secs_elapsed();

    /* Get number of tasks */ 
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);

    rank0_printf_timestamp("Starting %s MPI exercisers  %i tasks %s\n", location, numtasks, VARIANT_DESC);

    /* Synchronize printf timestamps across MPI ranks */
    last_printf_timestamp= secs_elapsed();

    /* Some tests break with sub communicator, so create one for use */
    if (MPI_Comm_dup(MPI_COMM_WORLD, &subcomm) != MPI_SUCCESS)
        punt("mpi_exerciser_driver: Failure in MPI_Comm_dup!\n");
      
    /* Do bcasts that usually use shared memory, useful for testing
     * cases where executable is not readable (chmod -r ./a.out)
     */
    do_bcast_noshmmem_test (8192, 131072, subcomm);

    /* This allreduce test has broken several MPIs at scale.
     * Scaled up to bigger arrays to try to trigger
     * same issues even at small scales.
     */
    do_allreduce_exerciser (2, 8192, 131072, subcomm);

    /* Free sub communicator */
    MPI_Comm_free (&subcomm);

    /* Time all the tests for rough timing info */
    end_time = secs_elapsed();
    run_time = end_time - start_time;

    rank0_printf_timestamp("Finished %s MPI exercisers RUNTIME %.4lf s %i tasks %s\n", location, run_time, numtasks, VARIANT_DESC);
}
#endif /* defined(USE_MPI) && !defined(NO_MPI_EXERCISERS) */

#ifdef TARGET_GPU_OMP 
/* If using OpenMP offloading, make sure GPU works before doing test */
STATIC void verify_gpu_openmp(int gpu_id)
{
    /* If using GPU, make sure GPU OpenMP gpu offloading works before doing test */
    int runningOnGPU = 0;

    char mpi_desc[50]="";

#ifdef USE_MPI
    /* indicate MPI used */
    strcpy(mpi_desc, " (all tasks)");

    /* Sync all the MPI ranks before selecting GPU */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Selecting GPU %i as default device%s\n", gpu_id, mpi_desc); 

    /* Pick GPU to use to exercise selection call */
    omp_set_default_device(gpu_id);

#ifdef USE_MPI
    /* Sync all the MPI ranks before printing start message */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Launching OpenMP GPU test kernel%s\n", mpi_desc); 

    /* Test if GPU is available using OpenMP4.5 legal code */
#pragma omp target map(from:runningOnGPU)
    {
        if (omp_is_initial_device() == 0)
            runningOnGPU = 1;
    }

    /* If still running on CPU, GPU must not be available, punt */
    if (runningOnGPU != 1)
        punt ("ERROR: OpenMP GPU test kernel did NOT run on GPU %i %s", gpu_id, VARIANT_DESC);

#ifdef USE_MPI
    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Verified OpenMP target test kernel ran on GPU%s\n", mpi_desc);
}
#endif

#if defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP)  
/* If using OpenMP offloading, make sure GPU works before doing test */
STATIC void verify_gpu_hip(int gpu_id)
{
    char mpi_desc[50]="";
    double *first_test_GPU_data;

#ifdef USE_MPI
    /* indicate MPI used */
    strcpy(mpi_desc, " (all tasks)");

    /* Sync all the MPI ranks before selecting GPU */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Selecting GPU %i as default device%s\n", gpu_id, mpi_desc); 

    /* Pick GPU to use to exercise selection call */
    HIPCHECK(hipSetDevice(gpu_id));

#ifdef USE_MPI
    /* Sync all the MPI ranks before printing start message */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Launching HIP GPU hipMalloc test%s\n", mpi_desc); 

    /* Make sure GPU is not totally broken */
    HIPCHECK(hipMalloc(&first_test_GPU_data, sizeof(double)*4));
    HIPCHECK(hipFree(first_test_GPU_data));

#ifdef USE_MPI
    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Verified hipMalloc worked on GPU%s\n", mpi_desc);
}
#endif /* defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP) */

#if defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA)  
/* If using OpenMP offloading, make sure GPU works before doing test */
STATIC void verify_gpu_cuda(int gpu_id)
{
    char mpi_desc[50]="";
    double *first_test_GPU_data;

#ifdef USE_MPI
    /* indicate MPI used */
    strcpy(mpi_desc, " (all tasks)");

    /* Sync all the MPI ranks before selecting GPU */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Selecting GPU %i as default device%s\n", gpu_id, mpi_desc); 

    /* Pick GPU to use to exercise selection call */
    CUDACHECK(cudaSetDevice(gpu_id));

#ifdef USE_MPI
    /* Sync all the MPI ranks before printing start message */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Launching CUDA GPU cudaMalloc test%s\n", mpi_desc); 

    /* Make sure GPU is not totally broken */
    CUDACHECK(cudaMalloc(&first_test_GPU_data, sizeof(double)*4));
    CUDACHECK(cudaFree(first_test_GPU_data));

#ifdef USE_MPI
    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("Verified cudaMalloc worked on GPU%s\n", mpi_desc);
} 
#endif /* defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA) */

#if defined(TARGET_GPU_OMP) || defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP) || defined(TARGET_GPU_LAMBDA_CUDA) || defined(TARGET_GPU_CUDA)
/* Returns secs_elapsed after MPI barrier (if MPI) and printing desc to rank 0 */
STATIC double sync_starttime(const char *desc)
{
    double start_time;

#ifdef USE_MPI
    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("%s", desc);

    start_time=secs_elapsed();
    return (start_time);
}

/* Returns secs_elapsed before MPI barrier (if MPI) and printing desc to rank 0 */
STATIC double sync_endtime(const char *desc)
{
    double end_time;

    end_time=secs_elapsed();

#ifdef USE_MPI
    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    rank0_printf_timestamp("%s", desc);

    return (end_time);
}
#endif

#if defined(TARGET_GPU_HIP) || defined(TARGET_GPU_CUDA)
STATIC __global__ void rush_larsen_gpu_kernel(double* m_gate, const long nCells, const double* Vm) 
{
    long ii = blockIdx.x*blockDim.x + threadIdx.x;
    if (ii > nCells) { return; }

    /* Identical contents to the loop body below */
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
#endif /* defined(TARGET_GPU_HIP) || defined(TARGET_GPU_CUDA) */

#if defined(TARGET_GPU_LAMBDA_HIP) || defined(TARGET_GPU_LAMBDA_CUDA)  
/* For both HIP and CUDA, emulate the RAJA Forall loop that executes
 * the lambda captured loop body for every iteration in range
 */
template < typename LoopBody >
STATIC __global__ void emulateRajaForall(LoopBody loop_body, const long nCells) {
    long ii = blockIdx.x*blockDim.x + threadIdx.x;
    if (ii > nCells) { return; }
    loop_body(ii);
}
#endif /* defined(TARGET_GPU_LAMBDA_HIP) || defined(TARGET_GPU_LAMBDA_CUDA) */

/* Sets up and runs the doRushLarsen kernel 'iterations' times, 
 * allocating CPU arrays and perhaps GPU arrays to consume 
 * kernel_mem_used GBs of memory.   
 *
 * This polynomial is a fit to the dynamics of a small part of a cardiac
 * myocyte, specifically the fast sodium m-gate described here:
 * https://www.ncbi.nlm.nih.gov/pubmed/16565318
 *
 * Does exactly the same work on every cell.   Can scale from one cell
 * to filling entire memory.   Does use cell's value as input
 * to calculations.
 * 
 * Returns number of data check failures, returns 0 if all data checks out.
 */
extern "C" long RUSH_LARSEN_VARIANT (long iterations, double kernel_mem_used)
{
    double kernel_starttime,kernel_endtime, kernel_runtime;
    double transfer_starttime,transfer_endtime, transfer_runtime;
    long nCells;
    long status_point;
    long fail_count =0;

#ifdef USE_MPI
    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    /* To make interop performance easier to compare,
     * start this file's timers over every time called.
     */

    /* Reset this file's secs_elapsed() counter to 0 */
    get_base_time(get_raw_secs());

    /* Synchronize printf timestamps across MPI ranks */
    last_printf_timestamp= secs_elapsed();

    /* Print separator before and after output with function name*/
    rank0_printf_timestamp("--------------- Begin rush_larsen_" VARIANT_DESC " (timer zeroed) ---------------\n");


    /* For print niceness, make .00000001 lower bound on GB memory */
    if (kernel_mem_used < .00000001)
        kernel_mem_used = .00000001;

    /* Calculate nCells from target memory target */
    nCells = (long) ((kernel_mem_used * 1024.0 * 1024.0 * 1024.0) / (sizeof(double) * 2));

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
            
    /* Print what we are running */
    rank0_printf_timestamp("START Rush Larsen %ld %.8f  cells %ld  %s\n", iterations, kernel_mem_used, nCells, VARIANT_DESC); 
    rank0_printf_timestamp("%s\n", VERSION_STRING); 

    /* If using GPU, make sure GPU works before doing test */
#ifdef TARGET_GPU_OMP 
    verify_gpu_openmp(0);

#elif defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP)  
    verify_gpu_hip(0);

#elif defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA)  
    verify_gpu_cuda(0);
#endif


#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP) 
    /* Print OpenMP thread mapping, syncs and aggregates MPI ranks (if MPI mode) */
    print_thread_stats ("Initial OpenMP");
#endif


    rank0_printf_timestamp("Allocating and initializing kernel arrays\n");

    double* m_gate = (double*)calloc(nCells,sizeof(double));
    if (m_gate == NULL)
    {
        punt ("%s failed calloc m_gate",VARIANT_DESC);
    }
         
    double* Vm = (double*)calloc(nCells,sizeof(double));
    if (Vm == NULL)
    {
        punt ("%s failed calloc Vm", VARIANT_DESC);
    }

    /* No data transfer time if not using GPU */
    transfer_starttime=0.0;
    transfer_endtime=0.0;

#if defined(TARGET_GPU_OMP)
    transfer_starttime=sync_starttime("Starting omp data map of CPU arrays to GPU\n");
#pragma omp target enter data map(to: m_gate[:nCells])
#pragma omp target enter data map(to: Vm[:nCells])
    transfer_endtime=sync_endtime("Finished omp data map of CPU arrays to GPU\n");
#endif 

#if defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP)  
    double *gpu_m_gate, *gpu_Vm;
    sync_starttime("Starting hipMalloc of GPU arrays\n");
    HIPCHECK(hipMalloc(&gpu_m_gate, sizeof(double)*nCells));
    HIPCHECK(hipMalloc(&gpu_Vm, sizeof(double)*nCells));
    sync_endtime("Finished hipMalloc of GPU arrays\n");

    transfer_starttime=sync_starttime("Starting hipMemcpy of CPU arrays to GPU arrays\n");
    HIPCHECK(hipMemcpy(gpu_m_gate, m_gate, sizeof(double)*nCells, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(gpu_Vm, Vm, sizeof(double)*nCells, hipMemcpyHostToDevice));
    transfer_endtime=sync_endtime("Finished hipMemcpy of CPU arrays to GPU arrays\n");
#endif /* defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP) */

#if defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA)  
    double *gpu_m_gate, *gpu_Vm;
    sync_starttime("Starting cudaMalloc of GPU arrays\n");
    CUDACHECK(cudaMalloc(&gpu_m_gate, sizeof(double)*nCells));
    CUDACHECK(cudaMalloc(&gpu_Vm, sizeof(double)*nCells));
    sync_endtime("Finished cudaMalloc of GPU arrays\n");

    transfer_starttime=sync_starttime("Starting cudaMemcpy of CPU arrays to GPU arrays\n");
    CUDACHECK(cudaMemcpy(gpu_m_gate, m_gate, sizeof(double)*nCells, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(gpu_Vm, Vm, sizeof(double)*nCells, cudaMemcpyHostToDevice));
    transfer_endtime=sync_endtime("Finished cudaMemcpy of CPU arrays to GPU arrays\n");
#endif /* defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA) */

#if defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP) || defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA)
    dim3 gridSize, blockSize;
    blockSize.x=512; blockSize.y=1; blockSize.z=1;
    gridSize.x = (nCells + blockSize.x-1) / blockSize.x; gridSize.y=1; gridSize.z=1;
#endif /* defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP) || defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA */

    transfer_runtime=transfer_endtime-transfer_starttime;
            
    /* Do the iterations asked for plus 1 for warmup */
    for (long itime=0; itime<=iterations; itime++) {
        /* Print warmup message for 0th iteration */
        if (itime == 0) 
        {
            rank0_printf_timestamp("Launching warmup iteration (not included in kernel timings)\n");
        }
                
        /* Print status every 10% of iterations */
        else if (((itime-1) % status_point) == 0)
        {   
            if (itime==1)
            {
#ifdef USE_MPI
                rank0_printf_timestamp("Waiting for all MPI ranks to complete warmup\n");
                /* Wait for all MPI ranks to complete */
                MPI_Barrier (MPI_COMM_WORLD);
#endif
#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP) 
                /* Print OpenMP thread mapping, syncs and aggregates MPI ranks (if MPI mode) */
                print_thread_stats ("Post-warmup OpenMP");
#endif

#if defined(USE_MPI) && !defined(NO_MPI_EXERCISERS)
                /* Do some short MPI exercisers that have exposed issues before (if in MPI mode) */
                mpi_exerciser_driver("Post-warmup");

#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP) 
                /* Print OpenMP thread mapping after MPI to see if impacted */
                print_thread_stats ("Post-MPI OpenMP");
#endif /* defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP)  */
#endif /* defined(USE_MPI) && !defined(NO_MPI_EXERCISERS) */

            }
            if (itime == 1)
                rank0_printf_timestamp("Starting kernel timings for Rush Larsen %ld %.8f\n", iterations, kernel_mem_used);
            rank0_printf_timestamp("Starting iteration %6li\n", itime);
        }

        /* Start timer after warm-up iteration 0 */
        if (itime==1)
        {
            kernel_starttime=secs_elapsed();
        }
                
        /*
         * RUSH LARSEN KERNEL BEING TIMED START
         */

/* Variants below use loop (or loop body) in the function */
#if defined(TARGET_CPU_SERIAL) || defined(TARGET_CPU_OMP) || defined(TARGET_GPU_OMP) || defined(TARGET_GPU_LAMBDA_HIP) || defined (TARGET_GPU_LAMBDA_CUDA)

#if defined(TARGET_GPU_LAMBDA_HIP) || defined(TARGET_GPU_LAMBDA_CUDA)
        auto& m_gate = gpu_m_gate; /* Capture GPU pointer instead of CPU one */
        auto& Vm = gpu_Vm;         /* Capture GPU pointer instead of GPU one */
        auto loop_body = [=] __device__ (long ii) /* Capture loop body for execution below */
#if 0 /* Leave this tricky ; comment to make emacs indent logic behave */
            ; 
#endif

#else /* No lambda capture, have for loop and possibly OpenMP directives */

/* OpenMP directives if target needs them */
#if defined(TARGET_GPU_OMP)
#pragma omp target teams distribute parallel for 
#elif defined(TARGET_CPU_OMP)
#pragma omp parallel for 
#endif
        /* For loop for TARGET_CPU_SERIAL, TARGET_CPU_OMP, and TARGET_GPU_OMP */
        for (long ii=0; ii<nCells; ii++) 
#endif
        {
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
        }; /* This ';' required TARGET_GPU_LAMBDA_HIP && TARGET_GPU_LAMBDA_CUDA */
#endif /* defined(TARGET_CPU_SERIAL) || defined(TARGET_CPU_OMP) || defined(TARGET_GPU_OMP) || defined(TARGET_GPU_LAMBDA_HIP) || defined (TARGET_GPU_LAMBDA_CUDA) */

#if defined(TARGET_GPU_LAMBDA_HIP) 
        /* Launch captured loop body on GPU using HIP and the emulateRajaForall device routine */
        hipLaunchKernelGGL((emulateRajaForall<decltype(loop_body)>), dim3(gridSize), dim3(blockSize), 0, 0, loop_body, nCells);
        HIPCHECK(hipDeviceSynchronize());

#elif defined(TARGET_GPU_HIP) 
        /* Launch captured loop body on GPU using HIP and the emulateRajaForall device routine */
        hipLaunchKernelGGL(rush_larsen_gpu_kernel, dim3(gridSize), dim3(blockSize), 0, 0, gpu_m_gate, nCells, gpu_Vm);
        HIPCHECK(hipDeviceSynchronize());

#elif defined(TARGET_GPU_LAMBDA_CUDA) 
        /* Launch captured loop body on GPU using CUDA and the emulateRajaForall device routine */
        emulateRajaForall<decltype(loop_body)><<<gridSize, blockSize>>>(loop_body, nCells);
        CUDACHECK(cudaDeviceSynchronize());

#elif defined(TARGET_GPU_CUDA) 
        /* Launch captured loop body on GPU using HIP and the emulateRajaForall device routine */
        rush_larsen_gpu_kernel<<<gridSize, blockSize>>>(gpu_m_gate, nCells, gpu_Vm);
        CUDACHECK(cudaDeviceSynchronize());
#endif
        /*
         * RUSH LARSEN KERNEL BEING TIMED END
         */
    }

    /* Get time after all iterations */
    kernel_endtime=secs_elapsed();

    /* Calculate kernel runtime */
    kernel_runtime = kernel_endtime-kernel_starttime;

    rank0_printf_timestamp("Finished kernel timings for Rush Larsen %ld %.8f\n", iterations, kernel_mem_used);

#ifdef USE_MPI
    rank0_printf_timestamp("Waiting for all MPI ranks to complete calculations\n");
    /* Wait for all MPI ranks to complete */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP) 
    /* Print OpenMP thread mapping, syncs and aggregates MPI ranks (if MPI mode) */
    print_thread_stats ("Final OpenMP");
#endif

    /* Print kernel runtime stats, syncs and aggregates MPI rank (if MPI mode) */
    print_runtime_stats(iterations, kernel_mem_used, kernel_runtime, transfer_runtime);

    /* Transfer GPU m_gate kernel memory to CPU kernel memory for data checks */
#ifdef TARGET_GPU_OMP
#pragma omp target update from (m_gate[0:nCells])

#elif defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP)
    HIPCHECK(hipMemcpy(m_gate, gpu_m_gate, sizeof(double)*nCells, hipMemcpyDeviceToHost));

#elif defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA)
    CUDACHECK(cudaMemcpy(m_gate, gpu_m_gate, sizeof(double)*nCells, cudaMemcpyDeviceToHost));
#endif


    /* Do sanity and consistency checks on all of m_gate. Including cross-rank if in MPI mode.
     * Prints PASS or FAIL based on data check results
     * Returns fail count so can be returned to caller.
     */
    fail_count = data_check (m_gate, iterations, kernel_mem_used, nCells);

    /* Free kernel GPU Memory */
#ifdef TARGET_GPU_OMP
#pragma omp target exit data map(delete:m_gate[:nCells])
#pragma omp target exit data map(delete:Vm[:nCells])

#elif defined(TARGET_GPU_HIP) || defined(TARGET_GPU_LAMBDA_HIP)
    HIPCHECK(hipFree(gpu_Vm));
    HIPCHECK(hipFree(gpu_m_gate));

#elif defined(TARGET_GPU_CUDA) || defined(TARGET_GPU_LAMBDA_CUDA)
    CUDACHECK(cudaFree(gpu_Vm));
    CUDACHECK(cudaFree(gpu_m_gate));
#endif

    /* Free kernel CPU Memory */
    free(Vm);
    free(m_gate);

    rank0_printf_timestamp("DONE Freed memory %s\n", VARIANT_DESC);

    /* Print separator before and after output */
    rank0_printf_timestamp("----------------- End rush_larsen_" VARIANT_DESC " ---------------\n");
   
    /* Return number of data check failures */
    return (fail_count);
}


/* Main driver (single test) when not being used in interop test */
#ifndef NO_MAIN
int main(int argc, char* argv[]) 
{
    long max_iterations=1;
    double kernel_mem_used=0.0;
    int rank = 0; /* Rank will be 0 for the no MPI case */
    int fail_count = 0;

#ifdef USE_MPI
    int  rc;
    double mpi_init_start_time, mpi_init_end_time, mpi_init_run_time;

    mpi_init_start_time = secs_elapsed();
    rc = MPI_Init(&argc,&argv);
    if (rc != MPI_SUCCESS) {
        printf ("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    mpi_init_end_time = secs_elapsed();
    mpi_init_run_time = mpi_init_end_time -mpi_init_start_time;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif

    if (argc != 3)
    {
        if (rank == 0)
        {
            printf ("Usage: %s  Iterations  Kernel_GBs_used\n", argv[0]);
            printf ("\n");
#ifdef TARGET_CPU_SERIAL
            printf ("Measure serial baseline small: %s 100000 .00000001\n", argv[0]);
            printf ("Measure serial baseline large: %s    100 10\n", argv[0]);
#else
            printf ("Measure serial launch overhead:  env OMP_NUM_THREADS=1 %s 100000 .00000001\n", argv[0]);
            printf ("Measure launch overhead:         %s 100000 .00000001\n", argv[0]);
            printf ("Measure kernel performance:      %s    100 10\n", argv[0]);
#endif
            printf ("\n");
            printf ("%s\n", VERSION_STRING);
            printf ("\n");
            printf ("RUSH_LARSEN_VARIANT: " rushxstr(RUSH_LARSEN_VARIANT) "\n" );
            printf ("VARIANT_DESC: " VARIANT_DESC "\n");
#ifdef NO_MPI_EXERCISERS
            printf ("Compiled with -DNO_MPI_EXERCISERS\n");
#endif
#ifdef NO_THREAD_MAP
            printf ("Compiled with -DNO_THREAD_MAP\n");
#endif
#ifdef NO_STATIC
            printf ("Compiled with -DNO_STATIC\n");
#endif
            printf ("\n");
            printf ("Questions? Contact John Gyllenhaal (gyllenhaal1@llnl.gov)\n");
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        exit (1);
    }

    /* Get iteration count and target kernel memory used arguments */
    max_iterations = atol(argv[1]);
    kernel_mem_used=atof(argv[2]);


#ifdef USE_MPI
    /* Print MPI_Init start time if using mpi */
    rank0_printf_timestamp("MPI_Init time %.4lf s %s\n", mpi_init_run_time, VARIANT_DESC);
#endif

    /*
     * Do the rush larsen test with the specified configuration 
     */
    fail_count = RUSH_LARSEN_VARIANT (max_iterations, kernel_mem_used);

#ifdef USE_MPI
    MPI_Finalize();
#endif

    /* Return 1 if data checks failed */
    if (fail_count == 0)
        return(0);
    else
        return(1);
}
#endif

