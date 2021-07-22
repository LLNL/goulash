# Goulash rush_larsen_interop2_cpu_omp

This is a multi-file C++ and Fortran interoperability test mechanically generated from rush_larsen/Generator/rush_larsen_interop_template.cc, rush_larsen/Generator/rush_larsen_cxx_template.cc, and rush_larsen/Generator/rush_larsen_fort_template.F90.   All source code and the Makefile will be overwritten if rush_larsen/Generator/generate_source is run.  

This interop tests are designed to allow testing build systems and compiler compatibility using a synthetic application of various levels of complexity.  All rush larsen variants in this interop test are also available as individual tests.   Build those individual tests first when debugging issues.

The naming conventions indicate how many different compiler vendors are used in the test:  
_interop1         - One compiler collection used to compile all files
_interop1.5       - One GPU OpenMP/HIP compiler and one CPU OpenMP compiler
_interop2         - Two compiler collections used (only one Fortran compiler used)
_interop2.5       - Two GPU OpenMP/HIP compilers and one CPU OpenMP compiler used (only one Fortran compiler)

The name also indicates the scope of the interoperability test:
_cpu_omp          - CPU-only OpenMP tests used
_gpu_omp_hip      - GPU and GPU OpenMP and GPU HIP tests used
*_mpi             - MPI tests in every variant

The interop GPU tests measure available GPU memory between tests vi HIP allocations and frees in order to help detect leaks and held memory between tests.

At the time of release of this benchmark _interop2 and _interop2.5 did not yet work on most platforms for GPU code due to ABI incompatibilities, etc.   Providing a testcase so vendors can solve those compatibility problems is one of reasons for the Goulash interop benchmarks.   Many platforms support CPU-only interoperability at this time(usually by running in gcc compatibility mode).

The actual compiler families used as 'compiler1', 'compiler2', and 'compiler3' are selected when rush_larsen/Generator/generate_source is run.   

A mechanically generated makefile in the directory supports 'make' 'make check' and 'make clean'.  Use or modify the makedef files in rush_larsen/Generator with rush_larsen/Generator/generate_source to recreate these makefiles to target your system and compilers.

Example build (from LLNL's RZNEVADA on 7/21/21) of a CPU only MPI test with 2 compilers:

> make
/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.7-cce-12.0.1/bin/mpicrayftn -c '-DCOMPILERID="cce-12.0.1"'  -O3 -g -fopenmp rush_larsen_cpu_omp_mpi_fort_compiler1.F90 -o rush_larsen_cpu_omp_fort_compiler1_mpi.o
/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.7-rocmcc-4.2.0/bin/mpiamdclang++ -c -DCOMPILERID=rocmcc-4.2.0  -O3 -g -fopenmp rush_larsen_cpu_omp_mpi_compiler2.cc -o rush_larsen_cpu_omp_mpi_compiler2.o
/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.7-cce-12.0.1/bin/mpicrayCC -c -DCOMPILERID=cce-12.0.1  -O3 -fopenmp rush_larsen_cpu_omp_mpi_compiler1.cc -o rush_larsen_cpu_omp_mpi_compiler1.o
/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.7-cce-12.0.1/bin/mpicrayCC -c -DCOMPILERID=cce-12.0.1  -O3 -fopenmp rush_larsen_interop2_cpu_omp_mpi.cc -o rush_larsen_interop2_cpu_omp_mpi.o
/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.7-cce-12.0.1/bin/mpicrayCC  -DCOMPILERID=cce-12.0.1  -O3 -fopenmp rush_larsen_cpu_omp_fort_compiler1_mpi.o rush_larsen_cpu_omp_mpi_compiler2.o rush_larsen_cpu_omp_mpi_compiler1.o rush_larsen_interop2_cpu_omp_mpi.o  -lmpifort_cray -o rush_larsen_interop2_cpu_omp_mpi

Here is a sample usage info from the above variant (run executable with -h or no arguments):

> ./rush_larsen_interop2_cpu_omp_mpi 
Usage: ./rush_larsen_interop2_cpu_omp_mpi  Interop_iterations rush_larsen_iterations  Kernel_GBs_used  

Measure launch overheads:    ./rush_larsen_interop2_cpu_omp_mpi  10 100000 .00000001  
Measure compute performance: ./rush_larsen_interop2_cpu_omp_mpi  10 100 10  
Emulate target use case:     ./rush_larsen_interop2_cpu_omp_mpi 100 100 28  
  
Work in kernel directly proportional to Kernel_GBs_used.  
  
Version 2.0 RC1 (7/21/21)  
  
VARIANT_DESC: interop_cpu_omp_mpi [cce-12.0.1]  
  
Here is a sample output from two nodes of RZNEVADA using a very short run via 'make check':  
> make check  
srun -n 2 ./rush_larsen_interop2_cpu_omp_mpi 1 10 .01  
IOP Rank 0:   0.083 (0.083s): MPI_Init time 0.0826 s   tasks 2  
IOP Rank 0:   0.083 (0.000s): ========== Initiating interoperability tests 1 10 0.01000000 (interop_cpu_omp_mpi [cce-12.0.1]) ==========  
IOP Rank 0:   0.083 (0.000s): ==== Calling rush_larsen_cpu_omp_mpi_compiler1()  Iter 1 of 1 [from cce-12.0.1] ====  
  0:   0.000 (0.000s): --------------- Begin rush_larsen_cpu_omp_mpi_compiler1 [cce-12.0.1] (timer zeroed) ---------------  
  0:   0.000 (0.000s): START Rush Larsen 10 0.01000000  cells 671088  cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.000 (0.000s): Version 2.0 RC1 (7/21/21)  
  0:   0.000 (0.000s): Printing Initial OpenMP mapping (all tasks)  
  1:   0.046 (0.046s): Initial OpenMP Map 0-109,111-127 Threads 127 rznevada2  
  0:   0.049 (0.049s): Initial OpenMP Map 0-63,65-127 Threads 127 rznevada1  
  0:   0.050 (0.000s): THREADSTATS Initial OpenMP 2 tasks  min 127  avg 127.0  max 127  maxdiff 0.00%  cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.050 (0.000s): Allocating and initializing kernel arrays  
  0:   0.050 (0.000s): Launching warmup iteration (not included in kernel timings)  
  0:   0.084 (0.034s): Waiting for all MPI ranks to complete warmup  
  0:   0.084 (0.000s): Printing Post-warmup OpenMP mapping (all tasks)  
  1:   0.087 (0.002s): Post-warmup OpenMP Map 0-127 Threads 128 rznevada2  
  0:   0.087 (0.003s): Post-warmup OpenMP Map 0-22,24-127 Threads 127 rznevada1  
  0:   0.087 (0.000s): THREADSTATS Post-warmup OpenMP 2 tasks  min 127  avg 127.5  max 128  maxdiff 0.79%  cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.087 (0.000s): Starting Post-warmup MPI exercisers  2 tasks cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.089 (0.002s): MPI_Bcast shmem exerciser with   8192 chars, ints, and doubles 2 tasks  
  0:   0.091 (0.002s): MPI_Bcast shmem exerciser with  16384 chars, ints, and doubles 2 tasks  
  0:   0.091 (0.000s): MPI_Bcast shmem exerciser with  32768 chars, ints, and doubles 2 tasks  
  0:   0.092 (0.000s): MPI_Bcast shmem exerciser with  65536 chars, ints, and doubles 2 tasks  
  0:   0.094 (0.002s): MPI_Bcast shmem exerciser with 131072 chars, ints, and doubles 2 tasks  
  0:   0.095 (0.001s): MPI_Bcast exerciser total RUNTIME 0.0061 s  8192 - 131072 ints 2 tasks  
  0:   0.095 (0.000s): MPI_Allreduce exerciser with   8192 ints 2 iterations 2 tasks  
  0:   0.097 (0.002s): MPI_Allreduce exerciser with  16384 ints 2 iterations 2 tasks  
  0:   0.098 (0.000s): MPI_Allreduce exerciser with  32768 ints 2 iterations 2 tasks  
  0:   0.098 (0.001s): MPI_Allreduce exerciser with  65536 ints 2 iterations 2 tasks  
  0:   0.100 (0.001s): MPI_Allreduce exerciser with 131072 ints 2 iterations 2 tasks  
  0:   0.102 (0.003s): MPI_Allreduce exerciser total RUNTIME 0.0074 s  8192 - 131072 ints 2 iterations 2 tasks  
  0:   0.102 (0.000s): Finished Post-warmup MPI exercisers RUNTIME 0.0157 s 2 tasks cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.102 (0.000s): Printing Post-MPI OpenMP mapping (all tasks)  
  0:   0.105 (0.002s): Post-MPI OpenMP Map 0-127 Threads 128 rznevada1  
  0:   0.105 (0.000s): THREADSTATS Post-MPI OpenMP 2 tasks  min 128  avg 128.0  max 128  maxdiff 0.00%  cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.105 (0.000s): Starting kernel timings for Rush Larsen 10 0.01000000  
  0:   0.105 (0.000s): Starting iteration      1  
  1:   0.105 (0.002s): Post-MPI OpenMP Map 0-127 Threads 128 rznevada2  
  0:   0.107 (0.002s): Starting iteration      2  
  0:   0.109 (0.002s): Starting iteration      3  
  0:   0.111 (0.002s): Starting iteration      4  
  0:   0.112 (0.002s): Starting iteration      5  
  0:   0.114 (0.002s): Starting iteration      6  
  0:   0.116 (0.002s): Starting iteration      7  
  0:   0.118 (0.002s): Starting iteration      8  
  0:   0.120 (0.002s): Starting iteration      9  
  0:   0.122 (0.002s): Starting iteration     10  
  0:   0.124 (0.002s): Finished kernel timings for Rush Larsen 10 0.01000000  
  0:   0.124 (0.000s): Waiting for all MPI ranks to complete calculations  
  0:   0.124 (0.000s): Printing Final OpenMP mapping (all tasks)  
  0:   0.126 (0.002s): Final OpenMP Map 0-127 Threads 128 rznevada1  
  0:   0.126 (0.000s): THREADSTATS Final OpenMP 2 tasks  min 128  avg 128.0  max 128  maxdiff 0.00%  cpu_omp_mpi_compiler1 [cce-12.0.1]  
  1:   0.126 (0.002s): Final OpenMP Map 0-127 Threads 128 rznevada2  
  0:   0.126 (0.000s): Collecting and aggregating kernel runtimes across MPI ranks  
  0:   0.126 (0.000s): VARIATION kernel    1.71%  datatrans    0.00%  2 tasks  Rush Larsen 10 0.01000000  cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.126 (0.000s): MINSTATS   Rush Larsen 10 0.01000000  0.0186 s  1857.21 us/iter  0.000 s datatrans cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.126 (0.000s): AVGSTATS   Rush Larsen 10 0.01000000  0.0187 s  1873.10 us/iter  0.000 s datatrans cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.126 (0.000s): MAXSTATS   Rush Larsen 10 0.01000000  0.0189 s  1888.99 us/iter  0.000 s datatrans cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.126 (0.000s): RUSHSTATS  Rush Larsen 10 0.01000000  0.0189 s  1888.99 us/iter  0.000 s datatrans cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.126 (0.000s): Starting data check for sanity and consistency  
  0:   0.127 (0.001s): PASSED Data check 10 0.01000000  m_gate[0]=0.976324219401755 cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.127 (0.000s): DONE Freed memory cpu_omp_mpi_compiler1 [cce-12.0.1]  
  0:   0.127 (0.000s): ----------------- End rush_larsen_cpu_omp_mpi_compiler1 [cce-12.0.1] ---------------  
IOP Rank 0:   0.230 (0.147s): ==== Calling rush_larsen_cpu_omp_mpi_compiler2()  Iter 1 of 1 [from cce-12.0.1] ====  
  0:   0.000 (0.000s): --------------- Begin rush_larsen_cpu_omp_mpi_compiler2 [rocmcc-4.2.0] (timer zeroed) ---------------  
  0:   0.000 (0.000s): START Rush Larsen 10 0.01000000  cells 671088  cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.000 (0.000s): Version 2.0 RC1 (7/21/21)  
  0:   0.000 (0.000s): Printing Initial OpenMP mapping (all tasks)  
  0:   0.003 (0.003s): Initial OpenMP Map 0-127 Threads 128 rznevada1  
  1:   0.003 (0.003s): Initial OpenMP Map 0-127 Threads 128 rznevada2  
  0:   0.003 (0.000s): THREADSTATS Initial OpenMP 2 tasks  min 128  avg 128.0  max 128  maxdiff 0.00%  cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.003 (0.000s): Allocating and initializing kernel arrays  
  0:   0.006 (0.003s): Launching warmup iteration (not included in kernel timings)  
  0:   0.008 (0.002s): Waiting for all MPI ranks to complete warmup  
  0:   0.008 (0.000s): Printing Post-warmup OpenMP mapping (all tasks)  
  0:   0.010 (0.003s): Post-warmup OpenMP Map 0-127 Threads 128 rznevada1  
  1:   0.010 (0.003s): Post-warmup OpenMP Map 0-127 Threads 128 rznevada2  
  0:   0.010 (0.000s): THREADSTATS Post-warmup OpenMP 2 tasks  min 128  avg 128.0  max 128  maxdiff 0.00%  cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.010 (0.000s): Starting Post-warmup MPI exercisers  2 tasks cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.011 (0.000s): MPI_Bcast shmem exerciser with   8192 chars, ints, and doubles 2 tasks  
  0:   0.011 (0.000s): MPI_Bcast shmem exerciser with  16384 chars, ints, and doubles 2 tasks  
  0:   0.011 (0.000s): MPI_Bcast shmem exerciser with  32768 chars, ints, and doubles 2 tasks  
  0:   0.011 (0.000s): MPI_Bcast shmem exerciser with  65536 chars, ints, and doubles 2 tasks  
  0:   0.011 (0.000s): MPI_Bcast shmem exerciser with 131072 chars, ints, and doubles 2 tasks  
  0:   0.012 (0.000s): MPI_Bcast exerciser total RUNTIME 0.0012 s  8192 - 131072 ints 2 tasks  
  0:   0.012 (0.000s): MPI_Allreduce exerciser with   8192 ints 2 iterations 2 tasks  
  0:   0.012 (0.000s): MPI_Allreduce exerciser with  16384 ints 2 iterations 2 tasks  
  0:   0.012 (0.000s): MPI_Allreduce exerciser with  32768 ints 2 iterations 2 tasks  
  0:   0.013 (0.001s): MPI_Allreduce exerciser with  65536 ints 2 iterations 2 tasks  
  0:   0.014 (0.001s): MPI_Allreduce exerciser with 131072 ints 2 iterations 2 tasks  
  0:   0.017 (0.003s): MPI_Allreduce exerciser total RUNTIME 0.0053 s  8192 - 131072 ints 2 iterations 2 tasks  
  0:   0.017 (0.000s): Finished Post-warmup MPI exercisers RUNTIME 0.0065 s 2 tasks cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.017 (0.000s): Printing Post-MPI OpenMP mapping (all tasks)  
  0:   0.020 (0.003s): Post-MPI OpenMP Map 0-127 Threads 128 rznevada1  
  0:   0.020 (0.000s): THREADSTATS Post-MPI OpenMP 2 tasks  min 128  avg 128.0  max 128  maxdiff 0.00%  cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.020 (0.000s): Starting kernel timings for Rush Larsen 10 0.01000000  
  1:   0.020 (0.003s): Post-MPI OpenMP Map 0-127 Threads 128 rznevada2  
  0:   0.020 (0.000s): Starting iteration      1  
  0:   0.022 (0.002s): Starting iteration      2  
  0:   0.023 (0.002s): Starting iteration      3  
  0:   0.025 (0.002s): Starting iteration      4  
  0:   0.027 (0.002s): Starting iteration      5  
  0:   0.029 (0.002s): Starting iteration      6  
  0:   0.031 (0.002s): Starting iteration      7  
  0:   0.032 (0.002s): Starting iteration      8  
  0:   0.034 (0.002s): Starting iteration      9  
  0:   0.036 (0.002s): Starting iteration     10  
  0:   0.038 (0.002s): Finished kernel timings for Rush Larsen 10 0.01000000  
  0:   0.038 (0.000s): Waiting for all MPI ranks to complete calculations  
  0:   0.038 (0.000s): Printing Final OpenMP mapping (all tasks)  
  0:   0.041 (0.003s): Final OpenMP Map 0-127 Threads 128 rznevada1  
  1:   0.040 (0.002s): Final OpenMP Map 0-127 Threads 128 rznevada2  
  0:   0.041 (0.000s): THREADSTATS Final OpenMP 2 tasks  min 128  avg 128.0  max 128  maxdiff 0.00%  cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.041 (0.000s): Collecting and aggregating kernel runtimes across MPI ranks  
  0:   0.041 (0.000s): VARIATION kernel    0.33%  datatrans    0.00%  2 tasks  Rush Larsen 10 0.01000000  cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.041 (0.000s): MINSTATS   Rush Larsen 10 0.01000000  0.0183 s  1834.89 us/iter  0.000 s datatrans cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.041 (0.000s): AVGSTATS   Rush Larsen 10 0.01000000  0.0184 s  1837.90 us/iter  0.000 s datatrans cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.041 (0.000s): MAXSTATS   Rush Larsen 10 0.01000000  0.0184 s  1840.90 us/iter  0.000 s datatrans cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.041 (0.000s): RUSHSTATS  Rush Larsen 10 0.01000000  0.0184 s  1840.90 us/iter  0.000 s datatrans cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.041 (0.000s): Starting data check for sanity and consistency  
  0:   0.042 (0.001s): PASSED Data check 10 0.01000000  m_gate[0]=0.976324219401755 cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.042 (0.000s): DONE Freed memory cpu_omp_mpi_compiler2 [rocmcc-4.2.0]  
  0:   0.042 (0.000s): ----------------- End rush_larsen_cpu_omp_mpi_compiler2 [rocmcc-4.2.0] ---------------  
IOP Rank 0:   0.272 (0.042s): ==== Calling rush_larsen_cpu_omp_mpi_fort_compiler1()  Iter 1 of 1 [from cce-12.0.1] ====  
  0:   0.000 (0.000s):--------------- Begin rush_larsen_cpu_omp_mpi_fort_compiler1 [cce-12.0.1] (timer zeroed) ---------------  
  0:   0.000 (0.000s): START Rush Larsen 10 iters 671088 cells 0.01000000 GBs cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.000 (0.000s): Version 2.0 RC1 (7/21/21)  
  0:   0.000 (0.000s): Printing Initial OpenMP mapping (all tasks)  
  0:   0.003 (0.003s): Initial OpenMP Map 0-6, 8-22, 24-127 Threads 126 rznevada1  
  1:   0.003 (0.003s): Initial OpenMP Map 0-127 Threads 128 rznevada1  
  0:   0.003 (0.000s): THREADSTATS Initial OpenMP 2 tasks  min 126  avg 127.0  max 128  maxdiff 1.59% cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.003 (0.000s): Allocating and initializing kernel arrays  
  0:   0.006 (0.003s): Launching warmup iteration (not included in timings)  
  0:   0.009 (0.002s): Waiting for all MPI ranks to complete warmup  
  0:   0.009 (0.000s): Printing Post-warmup OpenMP mapping (all tasks)  
  0:   0.011 (0.003s): Post-warmup OpenMP Map 0-6, 8-127 Threads 127 rznevada1  
  0:   0.011 (0.000s): THREADSTATS Post-warmup OpenMP 2 tasks  min 127  avg 127.5  max 128  maxdiff 0.79% cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  1:   0.011 (0.003s): Post-warmup OpenMP Map 0-127 Threads 128 rznevada1  
  0:   0.011 (0.000s): Starting Post-warmup MPI exercisers  2 tasks cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.011 (0.000s): MPI_Bcast shmem exerciser with   8192 chars, ints, and doubles 2 tasks  
  0:   0.012 (0.001s): MPI_Bcast shmem exerciser with  16384 chars, ints, and doubles 2 tasks  
  0:   0.012 (0.000s): MPI_Bcast shmem exerciser with  32768 chars, ints, and doubles 2 tasks  
  0:   0.013 (0.000s): MPI_Bcast shmem exerciser with  65536 chars, ints, and doubles 2 tasks  
  0:   0.013 (0.001s): MPI_Bcast shmem exerciser with 131072 chars, ints, and doubles 2 tasks  
  0:   0.015 (0.001s): MPI_Bcast exerciser total RUNTIME 0.0034 s 8192 - 131072 ints 2 tasks  
  0:   0.015 (0.000s): MPI_Allreduce exerciser total with   8192 ints 2 iterations 2 tasks  
  0:   0.015 (0.000s): MPI_Allreduce exerciser total with  16384 ints 2 iterations 2 tasks  
  0:   0.015 (0.000s): MPI_Allreduce exerciser total with  32768 ints 2 iterations 2 tasks  
  0:   0.016 (0.001s): MPI_Allreduce exerciser total with  65536 ints 2 iterations 2 tasks  
  0:   0.018 (0.001s): MPI_Allreduce exerciser total with 131072 ints 2 iterations 2 tasks  
  0:   0.020 (0.003s): MPI_Allreduce exerciser total RUNTIME 0.0055 s 8192 - 131072 ints 2 iterations 2 tasks  
  0:   0.020 (0.000s): Finished Post-warmup MPI exercisers RUNTIME 0.0091 s 2 tasks cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.020 (0.000s): Printing Post-MPI OpenMP mapping (all tasks)  
  0:   0.023 (0.003s): Post-MPI OpenMP Map 0-6, 8-127 Threads 127 rznevada1  
  1:   0.023 (0.003s): Post-MPI OpenMP Map 0-127 Threads 128 rznevada1  
  0:   0.023 (0.000s): THREADSTATS Post-MPI OpenMP 2 tasks  min 127  avg 127.5  max 128  maxdiff 0.79% cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.023 (0.000s): Starting kernel timings for Rush Larsen 10 0.01000000  
  0:   0.023 (0.000s): Starting iteration      1  
  0:   0.025 (0.002s): Starting iteration      2  
  0:   0.027 (0.002s): Starting iteration      3  
  0:   0.029 (0.002s): Starting iteration      4  
  0:   0.031 (0.002s): Starting iteration      5  
  0:   0.033 (0.002s): Starting iteration      6  
  0:   0.035 (0.002s): Starting iteration      7  
  0:   0.037 (0.002s): Starting iteration      8  
  0:   0.039 (0.002s): Starting iteration      9  
  0:   0.040 (0.002s): Starting iteration     10  
  0:   0.042 (0.002s): Finished kernel timings for Rush Larsen 10 0.01000000  
  0:   0.042 (0.000s): Waiting for all MPI ranks to complete calculations  
  0:   0.042 (0.000s): Printing Final OpenMP mapping (all tasks)  
  0:   0.045 (0.003s): Final OpenMP Map 0-6, 8-127 Threads 127 rznevada1  
  1:   0.045 (0.003s): Final OpenMP Map 0-127 Threads 128 rznevada1  
  0:   0.045 (0.000s): THREADSTATS Final OpenMP 2 tasks  min 127  avg 127.5  max 128  maxdiff 0.79% cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.045 (0.000s): Collecting and aggregating kernel runtimes across MPI ranks  
  0:   0.045 (0.000s): VARIATION kernel   55.40%  datatrans    0.00% 2 tasks  Rush Larsen 10 0.01000000 cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.045 (0.000s): MINSTATS   Rush Larsen 10 0.01000000  0.0123 s  1234.43 us/iter  0.0000 s datatrans cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.045 (0.000s): AVGSTATS   Rush Larsen 10 0.01000000  0.0158 s  1576.36 us/iter  0.0000 s datatrans cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.045 (0.000s): MAXSTATS   Rush Larsen 10 0.01000000  0.0192 s  1918.29 us/iter  0.0000 s datatrans cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.045 (0.000s): RUSHSTATS  Rush Larsen 10 0.01000000  0.0192 s  1918.29 us/iter  0.0000 s datatrans cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.045 (0.000s): Starting data check for sanity and consistency  
  0:   0.047 (0.001s): PASSED Data check 10 0.01000000  m_gate[0]=0.976324219401755 cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.047 (0.000s): DONE Freed memory cpu_omp_mpi_fort_compiler1 [cce-12.0.1]  
  0:   0.047 (0.000s):--------------- End rush_larsen_cpu_omp_mpi_fort_compiler1 [cce-12.0.1] ---------------  
IOP Rank 0:   0.319 (0.047s): INTEROP_PASS: ========== Completed interoperability tests 1 10 0.01000000 (interop_cpu_omp_mpi [cce-12.0.1]) ==========  
      
