# Goulash rush_larsen_gpu_omp_mpi_fort

This is a single file Fortran 2003 test mechanically generated from rush_larsen/Generator/rush_larsen_fort_template.F90.   It will be overwritten if rush_larsen/Generator/generate_source is run.   The fortran was hand generated in an attempt to replicate the C++ functionality and results.

This test is designed to allow direct performance comparisons between naively written HIP/CUDA and OpenMP GPU offloading schemes in a variety of coding styles and languages of a * parameterized embarrassingly parallel Rush Larsen kernel. Also allows testing build systems (including SPACK) handling of complicated build situations that LLNL cares about.
  
Designed to create ~20 single file test variants where no -D options required to select the variant and no include files are needed.  Almost all code in this file is identical between variants (this is intentional).  MPI support is ifdefed out for non-MPI variants.

The naming convention of the variant copies is intended to indicate variant abilities:  
 _cpu_serial      - single threaded, no OpenMP, on CPU  
 _cpu_omp         - use OpenMP to spawn threads on CPU  
 _gpu_omp         - uses OpenMP to offload to GPU  
 _gpu_hip         - uses HIP to offload to AMD or Nvidia GPU  
 _gpu_lambda_hip  - RAJA-like lambda HIP variant  
 _gpu_cuda        - uses CUDA to offload to Nvidia GPU  
 _gpu_lambda_cuda - RAJA-like lambda CUDA variant  
 *_mpi            - uses and exercises MPI e.g. _gpu_omp_mpi  
 *_fort           - Fortran version e.g. _gpu_omp_mpi_fort  

A mechanically generated makefile in the directory supports 'make' 'make check' and 'make clean'.  Use or modify the makedef files in rush_larsen/Generator with rush_larsen/Generator/generate_source to recreate these makefiles to target your system and compilers.

Example build (from LLNL's RZNEVADA on 7/21/21):

make  
/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.7-cce-12.0.1/bin/mpicrayftn  '-DCOMPILERID="cce-12.0.1"'  -O3 -g -fopenmp -haccel=amd_gfx908 rush_larsen_gpu_omp_mpi_fort.F90   -o rush_larsen_gpu_omp_mpi_fort

Here is a sample usage info from the above variant (run executable with -h or no arguments):

 Usage: ./rush_larsen_gpu_omp_mpi_fort  Iterations  Kernel_GBs_used  
    
 Measure serial launch overhead:  env OMP_NUM_THREADS=1 ./rush_larsen_gpu_omp_mpi_fort 100000 .00000001  
 Measure launch overhead:         ./rush_larsen_gpu_omp_mpi_fort 100000 .00000001  
 Measure kernel performance:      ./rush_larsen_gpu_omp_mpi_fort    100 10  
    
 Version 2.0 RC1 (7/21/21)  
    
 VARIANT_DESC: gpu_omp_mpi_fort [cce-12.0.1]  
  
Here is a sample output from two nodes of RZNEVADA using a very short run via 'make check':  
srun -n 2 ./rush_larsen_gpu_omp_mpi_fort 10 .01  
  0:   0.000 (0.000s):--------------- Begin rush_larsen_gpu_omp_mpi_fort [cce-12.0.1] (timer zeroed) ---------------  
  0:   0.000 (0.000s): START Rush Larsen 10 iters 671088 cells 0.01000000 GBs gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.000 (0.000s): Version 2.0 RC1 (7/21/21)  
  0:   0.002 (0.001s): Selecting GPU 0 as default device (all tasks)  
  0:   0.341 (0.339s): Launching OpenMP GPU test kernel (all tasks)  
  0:   0.863 (0.522s): Verified OpenMP target test kernel ran on GPU (all tasks)  
  0:   0.863 (0.000s): Allocating and initializing kernel arrays  
  0:   0.867 (0.005s): Starting omp data map of CPU arrays to GPU  
  0:   0.871 (0.004s): Finished omp data map of CPU arrays to GPU  
  0:   0.871 (0.000s): Launching warmup iteration (not included in timings)  
  0:   0.871 (0.000s): Waiting for all MPI ranks to complete warmup  
  0:   0.871 (0.000s): Starting Post-warmup MPI exercisers  2 tasks gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.873 (0.002s): MPI_Bcast shmem exerciser with   8192 chars, ints, and doubles 2 tasks  
  0:   0.876 (0.003s): MPI_Bcast shmem exerciser with  16384 chars, ints, and doubles 2 tasks  
  0:   0.877 (0.000s): MPI_Bcast shmem exerciser with  32768 chars, ints, and doubles 2 tasks  
  0:   0.877 (0.000s): MPI_Bcast shmem exerciser with  65536 chars, ints, and doubles 2 tasks  
  0:   0.880 (0.003s): MPI_Bcast shmem exerciser with 131072 chars, ints, and doubles 2 tasks  
  0:   0.881 (0.001s): MPI_Bcast exerciser total RUNTIME 0.0080 s 8192 - 131072 ints 2 tasks  
  0:   0.881 (0.000s): MPI_Allreduce exerciser total with   8192 ints 2 iterations 2 tasks
  0:   0.884 (0.002s): MPI_Allreduce exerciser total with  16384 ints 2 iterations 2 tasks  
  0:   0.884 (0.000s): MPI_Allreduce exerciser total with  32768 ints 2 iterations 2 tasks
  0:   0.885 (0.001s): MPI_Allreduce exerciser total with  65536 ints 2 iterations 2 tasks  
  0:   0.886 (0.001s): MPI_Allreduce exerciser total with 131072 ints 2 iterations 2 tasks  
  0:   0.889 (0.003s): MPI_Allreduce exerciser total RUNTIME 0.0078 s 8192 - 131072 ints 2 iterations 2 tasks  
  0:   0.889 (0.000s): Finished Post-warmup MPI exercisers RUNTIME 0.0180 s 2 tasks gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.889 (0.000s): Starting kernel timings for Rush Larsen 10 0.01000000  
  0:   0.889 (0.000s): Starting iteration      1  
  0:   0.889 (0.000s): Starting iteration      2  
  0:   0.890 (0.000s): Starting iteration      3  
  0:   0.890 (0.000s): Starting iteration      4  
  0:   0.890 (0.000s): Starting iteration      5  
  0:   0.890 (0.000s): Starting iteration      6  
  0:   0.890 (0.000s): Starting iteration      7  
  0:   0.890 (0.000s): Starting iteration      8  
  0:   0.890 (0.000s): Starting iteration      9  
  0:   0.890 (0.000s): Starting iteration     10  
  0:   0.890 (0.000s): Finished kernel timings for Rush Larsen 10 0.01000000  
  0:   0.890 (0.000s): Waiting for all MPI ranks to complete calculations  
  0:   0.890 (0.000s): Collecting and aggregating kernel runtimes across MPI ranks  
  0:   0.890 (0.000s): VARIATION kernel    5.14%  datatrans   14.94% 2 tasks  Rush Larsen 10 0.01000000 gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.890 (0.000s): MINSTATS   Rush Larsen 10 0.01000000  0.0010 s  96.49 us/iter  0.0031 s datatrans gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.890 (0.000s): AVGSTATS   Rush Larsen 10 0.01000000  0.0010 s  98.97 us/iter  0.0033 s datatrans gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.890 (0.000s): MAXSTATS   Rush Larsen 10 0.01000000  0.0010 s  101.45 us/iter  0.0036 s datatrans gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.890 (0.000s): RUSHSTATS  Rush Larsen 10 0.01000000  0.0010 s  101.45 us/iter  0.0036 s datatrans gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.892 (0.000s): Starting data check for sanity and consistency  
  0:   0.893 (0.001s): PASSED Data check 10 0.01000000  m_gate[0]=0.976324219401755 gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.893 (0.000s): DONE Freed memory gpu_omp_mpi_fort [cce-12.0.1]  
  0:   0.893 (0.000s):--------------- End rush_larsen_gpu_omp_mpi_fort [cce-12.0.1] ---------------  
  
