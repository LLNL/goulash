# Goulash rush_larsen_gpu_cuda

This is a single file C++ test mechanically generated from rush_larsen/Generator/rush_larsen_cxx_template.cc.   It will be overwritten if rush_larsen/Generator/generate_source is run.  

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
/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.7-cce-12.0.1/bin/mpicrayCC  -DCOMPILERID=cce-12.0.1 -D__HIP_PLATFORM_AMD__ -I/opt/rocm-4.2.0/include -O3 --cuda-gpu-arch=gfx908 -std=c++11 --rocm-path=/opt/rocm-4.2.0 -x hip rush_larsen_gpu_lambda_hip_mpi.cc   -o rush_larsen_gpu_lambda_hip_mpi

Here is a sample usage info from the above variant (run executable with -h or no arguments):

./rush_larsen_gpu_lambda_hip_mpi   
Usage: ./rush_larsen_gpu_lambda_hip_mpi  Iterations  Kernel_GBs_used  
  
Measure serial launch overhead:  env OMP_NUM_THREADS=1 ./rush_larsen_gpu_lambda_hip_mpi 100000 .00000001  
Measure launch overhead:         ./rush_larsen_gpu_lambda_hip_mpi 100000 .00000001  
Measure kernel performance:      ./rush_larsen_gpu_lambda_hip_mpi    100 10  
  
Version 2.0 RC1 (7/21/21)  
  
RUSH_LARSEN_VARIANT: rush_larsen_gpu_lambda_hip_mpi  
VARIANT_DESC: gpu_lambda_hip_mpi [cce-12.0.1]  

Here is a sample output from two nodes of RZNEVADA using a very short run via 'make check':  
srun -n 2 ./rush_larsen_gpu_lambda_hip_mpi 10 .01  
  0:   0.211 (0.211s): MPI_Init time 0.2111 s gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   0.000 (0.000s): --------------- Begin rush_larsen_gpu_lambda_hip_mpi [cce-12.0.1] (timer zeroed) ---------------  
  0:   0.000 (0.000s): START Rush Larsen 10 0.01000000  cells 671088  gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   0.000 (0.000s): Version 2.0 RC1 (7/21/21)  
  0:   0.000 (0.000s): Selecting GPU 0 as default device (all tasks)  
  0:   0.190 (0.190s): Launching HIP GPU hipMalloc test (all tasks)  
  0:   1.997 (1.807s): Verified hipMalloc worked on GPU (all tasks)  
  0:   1.997 (0.000s): Allocating and initializing kernel arrays  
  0:   1.998 (0.000s): Starting hipMalloc of GPU arrays  
  0:   1.998 (0.000s): Finished hipMalloc of GPU arrays  
  0:   1.998 (0.000s): Starting hipMemcpy of CPU arrays to GPU arrays  
  0:   2.004 (0.006s): Finished hipMemcpy of CPU arrays to GPU arrays  
  0:   2.004 (0.000s): Launching warmup iteration (not included in kernel timings)  
  0:   2.006 (0.002s): Waiting for all MPI ranks to complete warmup  
  0:   2.006 (0.000s): Starting Post-warmup MPI exercisers  2 tasks gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.011 (0.005s): MPI_Bcast shmem exerciser with   8192 chars, ints, and doubles 2 tasks  
  0:   2.019 (0.007s): MPI_Bcast shmem exerciser with  16384 chars, ints, and doubles 2 tasks  
  0:   2.019 (0.000s): MPI_Bcast shmem exerciser with  32768 chars, ints, and doubles 2 tasks  
  0:   2.019 (0.000s): MPI_Bcast shmem exerciser with  65536 chars, ints, and doubles 2 tasks  
  0:   2.021 (0.003s): MPI_Bcast shmem exerciser with 131072 chars, ints, and doubles 2 tasks  
  0:   2.022 (0.001s): MPI_Bcast exerciser total RUNTIME 0.0114 s  8192 - 131072 ints 2 tasks  
  0:   2.022 (0.000s): MPI_Allreduce exerciser with   8192 ints 2 iterations 2 tasks  
  0:   2.025 (0.002s): MPI_Allreduce exerciser with  16384 ints 2 iterations 2 tasks  
  0:   2.025 (0.000s): MPI_Allreduce exerciser with  32768 ints 2 iterations 2 tasks  
  0:   2.026 (0.001s): MPI_Allreduce exerciser with  65536 ints 2 iterations 2 tasks  
  0:   2.027 (0.001s): MPI_Allreduce exerciser with 131072 ints 2 iterations 2 tasks  
  0:   2.030 (0.003s): MPI_Allreduce exerciser total RUNTIME 0.0072 s  8192 - 131072 ints 2 iterations 2 tasks  
  0:   2.030 (0.000s): Finished Post-warmup MPI exercisers RUNTIME 0.0238 s 2 tasks gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.030 (0.000s): Starting kernel timings for Rush Larsen 10 0.01000000  
  0:   2.030 (0.000s): Starting iteration      1  
  0:   2.030 (0.000s): Starting iteration      2  
  0:   2.030 (0.000s): Starting iteration      3  
  0:   2.030 (0.000s): Starting iteration      4  
  0:   2.030 (0.000s): Starting iteration      5  
  0:   2.030 (0.000s): Starting iteration      6  
  0:   2.030 (0.000s): Starting iteration      7  
  0:   2.030 (0.000s): Starting iteration      8  
  0:   2.030 (0.000s): Starting iteration      9  
  0:   2.030 (0.000s): Starting iteration     10  
  0:   2.030 (0.000s): Finished kernel timings for Rush Larsen 10 0.01000000  
  0:   2.030 (0.000s): Waiting for all MPI ranks to complete calculations  
  0:   2.030 (0.000s): Collecting and aggregating kernel runtimes across MPI ranks  
  0:   2.030 (0.000s): VARIATION kernel   14.36%  datatrans    1.85%  2 tasks  Rush Larsen 10 0.01000000  gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.030 (0.000s): MINSTATS   Rush Larsen 10 0.01000000  0.0006 s  62.61 us/iter  0.006 s datatrans gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.030 (0.000s): AVGSTATS   Rush Larsen 10 0.01000000  0.0007 s  67.10 us/iter  0.006 s datatrans gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.030 (0.000s): MAXSTATS   Rush Larsen 10 0.01000000  0.0007 s  71.60 us/iter  0.006 s datatrans gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.030 (0.000s): RUSHSTATS  Rush Larsen 10 0.01000000  0.0007 s  71.60 us/iter  0.006 s datatrans gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.032 (0.000s): Starting data check for sanity and consistency  
  0:   2.033 (0.001s): PASSED Data check 10 0.01000000  m_gate[0]=0.976324219401755 gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.033 (0.000s): DONE Freed memory gpu_lambda_hip_mpi [cce-12.0.1]  
  0:   2.033 (0.000s): ----------------- End rush_larsen_gpu_lambda_hip_mpi [cce-12.0.1] ---------------  
  
