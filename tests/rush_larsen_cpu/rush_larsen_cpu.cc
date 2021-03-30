/*
  Copyright (c) 2019-21, Lawrence Livermore National Security, LLC. and other
  Goulash project contributors LLNL-CODE-795383, All rights reserved.
  For details about use and distribution, please read LICENSE and NOTICE from
  the Goulash project repository: http://github.com/llnl/goulash
  SPDX-License-Identifier: BSD-3-Clause
*/

/* Designed to allow performance comparisons between naively written
 * HIP/CUDA and OpenMP GPU offloading. This OpenMP CPU (non offloading)
 * version should allow comparison of all CPU cores against a
 * single GPU performance (rush_larsen_hip.cc).
 *
 * May be compiled with or without a COMPILERID specified (put in benchmark output)
 * /opt/rocm-4.0.1/llvm/bin/clang++  -o rush_larsen_cpu -O3 "-DCOMPILERID=rocm-4.0.1" -fopenmp rush_larsen_cpu.cc
 * 
 * May want to compare to no OpenMP overhead at all:
 * /opt/rocm-4.0.1/llvm/bin/clang++  -o rush_larsen_serial -O3 "-DCOMPILERID=rocm-4.0.1" -DNOOMP rush_larsen_cpu.cc
 *
 * Run with no arguments for argument info:
 *   Usage: ./rush_larsen_cpu  Iterations  Kernel_GBs_used
 *
 *     Measure serial launch overhead:   env OMP_NUM_THREADS=1 ./rush_larsen_cpu 100000 .00000001
 *     Measure thread launch overhead:   ./rush_larsen_cpu 100000 .00000001
 *     Measure CPU threaded performance: ./rush_larsen_cpu    100 10
 * 
 * 
 * Rush Larsen CUDA/Openmp kernels written by Rob Blake (LLNL) Sept 2016.
 * 
 * Pulled code from print_openmp_mapping.c by John Gyllenhaal at
 * LLNL written June 2020 which was based on code by Edgar Leon at LLNL
 * 
 * Inline performance measurements added (nvprof not needed)
 * by John Gyllenhaal (gyllenhaal1@llnl.gov) at LLNL 11/10/20.
 *
 * V1.1 03/22/21 command line args, perf diffs, maps, checks return codes and answer
 * V1.0 11/10/20 initial release, hard coded inputs, no error checking
 */

/* Allow a OpenMP-free serial version to be compiled with -DNOOMP */
#ifndef NOOMP
#include <omp.h>
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

/* The maximum number of threads supported */
#define MAX_SIZE 1024
int map_array[MAX_SIZE+10];

/* Spawns OpenMP threads and prints cpu mappings.
 * Send rank -1 if not in MPI program.
 */
int print_openmp_mapping (const char *desc, int rank)
{
    char map_buf[10000];
    char host[1024]="unknown";
    char tag[1024]="";
    char env_str[50000]="";
    char *omp_proc_bind_str=NULL, *omp_places_str=NULL, *omp_num_threads_str=NULL;
    int num_threads=-1;
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

#ifndef NOOMP
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
    printf_timestamp ("%s%s Threads %i %s%s\n", desc, map_buf, num_threads, tag, env_str);
#else
    printf_timestamp ("%s Single Threaded (NO OpenMP) %s\n", desc, tag);
#endif

    return (0);
}

/* Sets up and runs the doRushLarsen CPU kernel on all the available cores,
 * 'iterations' times, allocating CPU arrays to consume CPU_mem_used GBs
 * of host memory.   The desc is expected to be used in multi-compiler 
 * version of this benchmark later.
 *
 * This polynomial is a fit to the dynamics of a small part of a cardiac
 * myocyte, specifically the fast sodium m-gate described here:
 * https://www.ncbi.nlm.nih.gov/pubmed/16565318
 *
 * Does exactly the same work on every cell.   Can scale from one cell
 * to filling entire host memory.   Does use cell's value as input
 * to calculations.
 */
void rush_larsen_cpu(long iterations, double CPU_mem_used, const char *desc)
{
    double kernel_starttime,kernel_endtime;
    long nCells;
    long status_point;

    /* For print niceness, make .00000001 lower bound on GB memory */
    if (CPU_mem_used < .00000001)
	CPU_mem_used = .00000001;

    /* Calculate nCells from target memory target */
    nCells = (long) ((CPU_mem_used * 1024.0 * 1024.0 * 1024.0) / (sizeof(double) * 2));

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
    printf_timestamp("START Rush Larsen %ld iters %ld cells %.8f GBs CPU%s\n", iterations, nCells, CPU_mem_used, desc); 
    printf_timestamp("%s\n", version_string); 

    /* Print OpenMP thread mapping, no MPI so pass in -1 */
    print_openmp_mapping ("Initial OpenMP ", -1);


    printf_timestamp("Allocating and initializing CPU arrays\n");
    double* m_gate = (double*)calloc(nCells,sizeof(double));
    if (m_gate == NULL)
    {
        punt (" CPU%s failed calloc m_gate",desc);
    }
 
    double* Vm = (double*)calloc(nCells,sizeof(double));
    if (Vm == NULL)
    {
        punt ("CPU%s failed calloc Vm", desc);
    }
    
    /* Do the iterations asked for plus 1 for warmup */
    for (int itime=0; itime<=iterations; itime++) {
        /* Print warmup message for 0th iteration */
	if (itime == 0)
	    printf_timestamp("Launching warmup iteration (not included in timings)\n");
	
        /* Print status every 10% of iterations */
	else if (((itime-1) % status_point) == 0)
        {   
	    if (itime==1)
            {
                 /* Print OpenMP thread mapping, no MPI so pass in -1 */
                 print_openmp_mapping ("Post-warmup OpenMP ", -1);
            }
	    printf_timestamp("Starting iteration %6li\n", itime);
        }

        /* Start timer after warm-up iteration 0 */
	if (itime==1)
	    kernel_starttime=secs_elapsed();
        
#ifndef NOOMP
#pragma omp parallel for
#endif
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
	}
    }

    /* Get time after all iterations */
    kernel_endtime=secs_elapsed();

    /* Print time stats */
    printf_timestamp("STATS Rush Larsen %li iters %.4lf sec %.2lf us/iter CPU%s\n",  iterations, kernel_endtime-kernel_starttime, (double)(kernel_endtime-kernel_starttime)*1000000.0/(double) iterations, desc);

    /* Sanity check that CPU not giving garbage
     * Found m_gate[0] to be ~.0.506796353074569 after 1 iteration (really 2 with warmup)
     * and converges to 0.996321172062538 after 100 iterations.  Make sure in that bounds
     * for now.  With a little slop (~.000001) for now (not sure rounding error expected)
     */
    if (m_gate[0] < 0.506796) 
    {
	printf_timestamp("FAILED Data sanity check m_gate[0]=%.15lf < 0.506796 (0.506796353074569 min expected value) CPU%s\n", m_gate[0], desc);
    }
    else if (m_gate[0] > 0.996322)
    {
	printf_timestamp("FAILED Data sanity check m_gate[0]=%.15lf > 0.996322 (0.996321172062538 max expected value) CPU%s\n", m_gate[0], desc);
    }
    else
    {
	printf_timestamp("PASSED Data sanity check m_gate[0]=%.15lf CPU%s\n", m_gate[0], desc);
    }

    /* Print OpenMP thread mapping, no MPI so pass in -1 */
    print_openmp_mapping ("Final OpenMP ", -1);

    /* Free CPU Memory */
    free(Vm);
    free(m_gate);

    printf_timestamp("DONE Freed memory CPU%s\n", desc);
}

int main(int argc, char* argv[]) 
{
    long max_iterations=1;
    double CPU_mem_used=0.0;
    char tag[100];

    if (argc != 3)
    {
        printf ("Usage: %s  Iterations  Kernel_GBs_used\n", argv[0]);
        printf ("\n");
        printf ("Measure serial launch overhead:   env OMP_NUM_THREADS=1 %s 100000 .00000001\n", argv[0]);
        printf ("Measure thread launch overhead:   %s 100000 .00000001\n", argv[0]);
        printf ("Measure CPU threaded performance: %s    100 10\n", argv[0]);
        printf ("\n");
        printf ("%s\n", version_string);
        printf ("\n");
        printf ("Questions? Contact John Gyllenhaal (gyllenhaal1@llnl.gov)\n");
        exit (1);
    }

    /* Get iteration count and target CPU memory used arguments */
    max_iterations = atol(argv[1]);
    CPU_mem_used=atof(argv[2]);

    /* The CODETAG defined indirectly by -DCOMPILERID on compile line */
#ifdef COMPILERID
#define rushxstr(s) rushstr(s)
#define rushstr(s) #s
    snprintf(tag, sizeof(tag), " [%s]", rushxstr(COMPILERID));
#else
    strcpy(tag, "");
#endif
   
    rush_larsen_cpu(max_iterations, CPU_mem_used, tag);

    return(0);
}

