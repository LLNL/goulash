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
 * The Goulash project conceived of and designed by David Richards,
 * Tom Scogland, and John Gyllenhaal at LLNL Oct 2019.
 * Please contact John Gyllenhaal (gyllenhaal1@llnl.gov) with questions.
 *
 * Interop versions create by John Gyllenhaal at LLNL 07/14/21
 * to test mixing all the Rush Larsen tests with multiple GPU compilers
 * all in one final executable.
 *  
 * Initial test generator from template files, including Makefiles
 * created by John Gyllenhaal at LLNL 07/21/21 for V2.0RC1
 *
 * V2.0 RC1 07/21/21 Added MPI support, interop version, enhanced data checks.
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
 * Unlocking designed to allow regression testing on all supported variants
 * from one source file before copying and hard coding these settings
 * for each variant when pushing out a new Goulash version.
 */
#ifndef GOULASH_UNLOCK

/* MUST BE EXACTLY TWO SPACES BETWEEN #undef AND VARIABLE */
/* THIS IS REQUIRED FOR THE TEMPLATE INSTANTIATION */

/* Exactly one TARGET must be defined to determine variant used */
#undef  TARGET_CPU_SERIAL
#undef  TARGET_CPU_OMP
#undef  TARGET_GPU_OMP
#undef  TARGET_GPU_HIP
#undef  TARGET_GPU_OMP_HIP
#undef  TARGET_GPU_CUDA
#undef  TARGET_GPU_OMP_CUDA
 
/* Use MPI variant */
#undef  USE_MPI

/* Make all support routines static scope */
#undef  NOSTATIC

#endif /* !GOULASH_UNLOCK */

/* nvcc does not like these preprocessor directives */
#ifdef __CUDA__
/*
 * Make sure exactly one TARGET is defined
 * Generate preprocessing error otherwise.
 */
#if !((defined(TARGET_CPU_SERIAL) + defined(TARGET_CPU_OMP) + defined(TARGET_GPU_OMP) + defined(TARGET_GPU_HIP) + defined(TARGET_GPU_OMP_HIP) + defined(TARGET_GPU_CUDA) + defined(TARGET_GPU_OMP_CUDA)) == 1)
/* Do not have exactly one target
 * Indicate if zero targets or multiple targets is the problem
 */
#if ((defined(TARGET_CPU_SERIAL) + defined(TARGET_CPU_OMP) + defined(TARGET_GPU_OMP) + defined(TARGET_GPU_HIP) + defined(TARGET_GPU_OMP_HIP) + defined(TARGET_GPU_CUDA) + defined(TARGET_GPU_OMP_CUDA)) == 0)
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
#ifdef TARGET_GPU_OMP_HIP
#error Multiple targets specified: TARGET_GPU_OMP_HIP
#endif
#ifdef TARGET_GPU_CUDA
#error Multiple targets specified: TARGET_GPU_CUDA
#endif
#ifdef TARGET_GPU_OMP_CUDA
#error Multiple targets specified: TARGET_GPU_OMP_CUDA
#endif
#endif
#endif

/* If NOSTATIC defined, make all support routines non-static (visible) */
#ifdef NOSTATIC
#define STATIC
#else
#define STATIC static
#endif

/* Preprocessor macro rushglue(x,y) glues two defined values together */
#define rushglue2(x,y) x##y
#define rushglue(x,y) rushglue2(x,y)

/* Proprocessor macro rushxstr(s) converts value to string */
#define rushxstr2(s) #s
#define rushxstr(s) rushxstr2(s)


/* Create unique VARIANT_TAG based on file #def and #undef
 * settings that is used to create rush_larsen function
 * call name and to annotate key lines of output
 */
#undef TARGET_TAG
#if   defined(TARGET_CPU_SERIAL)
#define TARGET_TAG interop_cpu_serial
#elif defined(TARGET_CPU_OMP)
#define TARGET_TAG interop_cpu_omp
#elif defined(TARGET_GPU_OMP)
#define TARGET_TAG interop_gpu_omp
#elif defined(TARGET_GPU_HIP)
#define TARGET_TAG interop_gpu_hip
#elif defined(TARGET_GPU_OMP_HIP)
#define TARGET_TAG interop_gpu_omp_hip
#elif defined(TARGET_GPU_CUDA)
#define TARGET_TAG interop_gpu_cuda
#elif defined(TARGET_GPU_OMP_CUDA)
#define TARGET_TAG interop_gpu_omp_cuda
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

/* In case we want to use a VARIANT_ID later, leave logic in */
#define VARIANT_ID
#undef VARIANT_TAG
#define VARIANT_TAG rushglue(RUSH_MPI_TAG,VARIANT_ID)

/* Generate VARIANT_DESC string that annotates the end of key output
 * lines spread across this whole file.  Uses C trick that
 * "omp" " [" "g++" "]"
 * is equivalent to
 * "omp [g++]"
 * Since I could not figure out how to create one big string
 * with preprocessor.
 */
#ifdef COMPILERID
#define VARIANT_DESC rushxstr(VARIANT_TAG) " ["  rushxstr(COMPILERID) "]"
#define INTEROP_COMPILER "[from "  rushxstr(COMPILERID) "] "
#else
#define VARIANT_DESC rushxstr(VARIANT_TAG)
#define INTEROP_COMPILER ""
#endif


#if defined(TARGET_GPU_HIP) || defined(TARGET_GPU_OMP_HIP)
#include "hip/hip_runtime.h"
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
#ifdef USE_MPI
#include "mpi.h"
#endif

/* Returns time in seconds (double) since the first call to secs_elapsed
 * (i.e., the first call returns 0.0).
 */
STATIC double secs_elapsed( void )
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

/* Works like vfprintf, except prefixes wall-clock time (using secs_elapsed)
 * and the difference since last vfprintf.
 * Also flushes out after printing so messages appear immediately
 * Used to implement printf_interop, punt, etc.
 */
STATIC double last_printf_interop=0.0;
STATIC void vfprintf_interop (FILE*out, const char * fmt, va_list args)
{
    char buf[4096];
    int rank = -1;  /* Don't print rank for serial runs */
#ifdef USE_MPI
    /* Get actual MPI rank if using MPI */
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif

    /* Get wall-clock time since first call to secs_elapsed */
    double sec = secs_elapsed();
    double diff = sec - last_printf_interop;
    last_printf_interop=sec;

    /* Print out passed message to big buffer*/
    vsnprintf (buf, sizeof(buf), fmt, args);

    /* No MPI case */
    if (rank < 0)
    {
        /* Print out timestamp and diff seconds with buffer*/
        fprintf (out, "IOP: %7.3f (%05.3fs): %s", sec, diff, buf);
    }
    /* MPI case, add rank */
    else
    {
        /* Print out timestamp and diff seconds and MPI rank with buffer*/
        fprintf (out, "IOP Rank %i: %7.3f (%05.3fs): %s", rank, sec, diff, buf);
    }

    /* Flush out, so message appears immediately */
    fflush (out);
}

/* Prints to stdout for all MPI ranks with timestamps and time diffs */
STATIC void printf_interop (const char * fmt, ...)
{
    va_list args;
    va_start (args, fmt);

    /* Use helper routine to actually do print and flush */
    vfprintf_interop (stdout,fmt,args);

    va_end (args);
}

STATIC void punt_interop (const char * fmt, ...)
{
    va_list args;
    va_start (args, fmt);

    /* Flush stdout, so pending message appears before punt message */
    fflush (stdout);

    /* Use helper routine to actually do print and flush */
    vfprintf_interop (stderr,fmt,args);

    va_end (args);

    /* Abort the program */
    exit(1);
}



#if defined(TARGET_GPU_HIP) || defined(TARGET_GPU_OMP_HIP)
/* Pulled from Goulash's openmp_mem_usage_hip.cc */
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
        punt_interop ("GPU %i failed hipSetDevice: %s", gpu_id, hipGetErrorString(err));
    }

    /* Make sure GPU is not totally broken before doing test */
    double* dummyfirst;
    err=hipMalloc(&dummyfirst, sizeof(double)*4);
    if (err != hipSuccess)
    {
        punt_interop ("GPU %i failed first hipMalloc: %s", gpu_id, hipGetErrorString(err));
    }
    err=hipFree(dummyfirst);
    if (err != hipSuccess)
    {
        punt_interop ("GPU %i failed first hipFree: %s", gpu_id,  hipGetErrorString(err));
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
        printf_interop("WARNING: May have measured less then full gpu memory!   MAX_1GB_ALLOCS (%i) hit!\n", MAX_1GB_ALLOCS);

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
        printf_interop("WARNING: GPU memory may be fragmented, more than 1GB of 16MB allocs (%4.2lf GBs) allocated! \n", ((index_16MB*16.0)/1024.0) );


    if (index_16MB >= MAX_16MB_ALLOCS)
        printf_interop("WARNING: May have measured less then full gpu memory!   MAX_16MB_ALLOCS (%i) hit!\n", MAX_16MB_ALLOCS);

    /* Free memory in reverse order allocated */
    for (int i=index_16MB-1; i >= 0; i--)
    {
        err=hipFree(mem_16MB_array[i]);
        if (err != hipSuccess)
        {
            punt_interop ("GPU %i failed hipFree 16MB alloc %i : %s", gpu_id, i, hipGetErrorString(err));
        }
    }

    for (int i=index_1GB-1; i >= 0; i--)
    {
        err=hipFree(mem_1GB_array[i]);
        if (err != hipSuccess)
        {
            punt_interop ("GPU %i failed hipFree 1GB alloc %i : %s", gpu_id, i, hipGetErrorString(err));
        }
    }

    /* Make it easier to print out diffs of memory measurements */
    diff_last_gpu_mem = alloced_mem-last_gpu_mem;
    last_gpu_mem = alloced_mem;

    return (alloced_mem);
}
#endif /* defined(TARGET_GPU_HIP) || defined(TARGET_GPU_OMP_HIP) */

STATIC void print_mem_stats ( const char *desc, int iter, int max_iter)
{
#if defined(TARGET_GPU_HIP) || defined(TARGET_GPU_OMP_HIP) 
    double gpu_mem;
    int rank = 0; /* Rank will be 0 for the no MPI case */
    int numtasks = 1; /* Will be 1 for no MPI case */
    char iter_desc[100] ;

#ifdef USE_MPI 
    double gpu_maxmem, gpu_minmem, gpu_summem, gpu_avgmem, gpu_variation;
    int ret_code;
   
    /* Sync all the MPI ranks before starting */
    MPI_Barrier (MPI_COMM_WORLD);
#endif

    /* Create iteration description for all output */
    if (iter >= 1)
        sprintf (iter_desc, "  Iter %i of %i", iter, max_iter);
    else
        sprintf (iter_desc, "");

#ifdef USE_MPI 
    /* Get actual MPI rank if using MPI */
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    /* Get number of tasks */
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
#endif

    if (numtasks > 1)
    {
        if (rank == 0)
        {
            printf_interop ("Measuring free GPU memory (all tasks) %s%s\n", desc, iter_desc);
        }
    }
    else
    {
        printf_interop ("Measuring free GPU memory %s%s\n", desc, iter_desc);
    }

    /* Get free GPU memory from every rank */
    gpu_mem =  measure_gpu_mem (0);


#ifdef USE_MPI
    ret_code = MPI_Allreduce (&gpu_mem, &gpu_maxmem, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt_interop ("Error: MPI_Allreduce max mem returned %i\n", ret_code);

    ret_code = MPI_Allreduce (&gpu_mem, &gpu_minmem, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt_interop ("Error: MPI_Allreduce min mem returned %i\n", ret_code);

    ret_code = MPI_Allreduce (&gpu_mem, &gpu_summem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (ret_code != MPI_SUCCESS)
        punt_interop ("Error: MPI_Allreduce sum mem returned %i\n", ret_code);
    gpu_avgmem = gpu_summem / (double) numtasks;
    if (gpu_minmem > 0.0001)
        gpu_variation = (gpu_maxmem - gpu_minmem)*100.0/gpu_minmem;
    else
        gpu_variation = (gpu_maxmem - gpu_minmem)*100.0/0.0001;

    if ((rank == 0) && (numtasks > 1))
    {
        printf_interop ("VARGPUMEM: %7.2lf%% GPU mem variation %s%s\n", gpu_variation, desc, iter_desc);
        printf_interop ("MINGPUMEM:  %6.2lf gb free GPU memory %s%s\n", gpu_minmem, desc, iter_desc);
        printf_interop ("AVGGPUMEM:  %6.2lf gb free GPU memory %s%s\n", gpu_avgmem, desc, iter_desc);
        printf_interop ("MAXGPUMEM:  %6.2lf gb free GPU memory %s%s\n", gpu_maxmem, desc, iter_desc);
    }

    /* For overall stats, use min memory available as gpu_mem */
    gpu_mem = gpu_minmem;
#endif


    if (rank == 0)
    { 
        /* Print overall stats  */
        printf_interop ("GPUMEMFREE: %6.2lf gb free GPU memory %s%s\n", gpu_mem, desc, iter_desc);
    }
#endif /* defined(TARGET_GPU_HIP) || defined(TARGET_GPU_OMP_HIP) */
}   

#define CXXPROTO(x)  extern "C" long x (long iterations, double kernel_mem_used)
#define FORTPROTO(x) extern "C" long x##_ (long &iterations, double &kernel_mem_used)

/* 
 * Prototype all the variants we expect to link to for all the compilers 
 */
 
#undef  ADD_PROTOTYPES_ABOVE_HERE
/* Above line used by the Generator/generate_source script to add prototypes */

/* Main driver for interop test which will call all of the above variants */
int main(int argc, char* argv[]) 
{
    long interop_iterations=1, rush_iterations=1;
    double kernel_mem_used=0.0;
    int rank = 0; /* Rank will be 0 for the no MPI case */
    int fail_count = 0;
    int iter;

#ifdef USE_MPI
    int  rc;
    double mpi_init_start_time, mpi_init_end_time, mpi_init_run_time;
    int numtasks;


    mpi_init_start_time = secs_elapsed();
    rc = MPI_Init(&argc,&argv);
    if (rc != MPI_SUCCESS) {
        printf ("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    mpi_init_end_time = secs_elapsed();
    mpi_init_run_time = mpi_init_end_time -mpi_init_start_time;

    /* Get rank */
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    /* Get number of tasks */
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
#endif

    if (argc != 4)
    {
        if (rank == 0)
        {
            printf ("Usage: %s  Interop_iterations rush_larsen_iterations  Kernel_GBs_used\n", argv[0]);
            printf ("\n");
            printf ("Measure launch overheads:    %s  10 100000 .00000001\n", argv[0]);
            printf ("Measure compute performance: %s  10 100 10\n", argv[0]);
            printf ("Emulate target use case:     %s 100 100 28\n", argv[0]);
            printf ("\n");
            printf ("Work in kernel directly proportional to Kernel_GBs_used.\n");
            printf ("\n");
            printf ("%s\n", VERSION_STRING);
            printf ("\n");
            printf ("VARIANT_DESC: %s\n", VARIANT_DESC);
            printf ("\n");
            printf ("Questions? Contact John Gyllenhaal (gyllenhaal1@llnl.gov)\n");
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        exit (1);
    }

    /* Get interop and rush iteration counts and target CPU memory used arguments */
    interop_iterations = atol(argv[1]);
    rush_iterations = atol(argv[2]);
    kernel_mem_used=atof(argv[3]);

    /* Make arguments sane if garbage input */
    if (interop_iterations < 1)
        interop_iterations = 1;

    if (rush_iterations < 1)
        rush_iterations = 1;

    if (kernel_mem_used < .00000001)
        kernel_mem_used = .00000001; 


#ifdef USE_MPI
    if (rank == 0)
    {
        /* Print MPI_Init start time if using mpi */
        printf_interop("MPI_Init time %.4lf s   tasks %i\n", mpi_init_run_time, numtasks);
    }
#endif
   
    if (rank == 0) printf_interop("========== Initiating interoperability tests %li %li %.8lf (%s) ==========\n", interop_iterations, rush_iterations, kernel_mem_used, VARIANT_DESC);

    print_mem_stats ("before calling any test", 0, 0);

#define CXXLAUNCH(x) if (rank == 0) printf_interop("==== Calling %s()  Iter %i of %i %s====\n", rushxstr( x ),  iter, interop_iterations, INTEROP_COMPILER);  fail_count += x (rush_iterations, kernel_mem_used); print_mem_stats("after calling " rushxstr( x) "()", iter, interop_iterations )

#define FORTLAUNCH(x) if (rank == 0) printf_interop("==== Calling %s()  Iter %i of %i %s====\n", rushxstr( x ),  iter, interop_iterations, INTEROP_COMPILER); fail_count += x##_ (rush_iterations, kernel_mem_used); print_mem_stats("after calling " rushxstr( x) "()", iter, interop_iterations )

    /* Interop needs to iterate over all the tests (ideally many times) */
    for (iter=1; iter <= interop_iterations; iter++)
    {
        /*
         * Do the rush larsen tests with the specified configurations 
         */
#undef  ADD_LAUNCH_CALLS_ABOVE_HERE
/* Above line used by the Generator/generate_source script to add launch calls */
    }

    /* Print message and return 1 if data correction errors detected */
    if (fail_count != 0)
    {
        if (rank == 0) printf_interop("INTEROP_FAIL: ERROR: Interop tests failed, detected %i data check errors\n", fail_count);

#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        exit(1);
    }

    if (rank == 0) 
    {  
        printf_interop("INTEROP_PASS: ========== Completed interoperability tests %li %li %.8lf (%s) ==========\n", interop_iterations, rush_iterations, kernel_mem_used, VARIANT_DESC);
       
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    if (fail_count == 0)
        return(0);
    else
        return(1);
}

