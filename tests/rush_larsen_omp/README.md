# Goulash rush_larsen_omp

This test (no MPI, OpenMP-kernels using GPU) is one of the building blocks we are using
for interoperability tests (building with multiple compilers in the same
application). An interop version will be added soon to goulash.
Building and verifying the goulash components separately is recommmended.

This OpenMP-based GPU benchmark should be able to match the performance
of the HIP-based GPU benchmark rush_larsen_hip from Goulash.

Note: With this initial release, we provided a trivial Makefile using 
ROCM to get you started but you will likely have to figure
out your build line manually for now.   Providing relevant testcases 
for build systems in order to figure out how to build these complex 
cases is one of the goals of goulash.

Here is a sample usage info (run executable with -h or no arguments):

Usage: ./rush_larsen_omp  Iterations  Kernel_GBs_used

Measure launch overhead: ./rush_larsen_omp 100000 .00000001
Measure GPU performance: ./rush_larsen_omp    100 10

Version 1.1 (3/22/21)

Here is a sample benchmark output from a single AMD MI60 GPU (rocm-4.0.1):
./rush_larsen_omp 100 10
  0.000 (0.000s): START Rush Larsen 100 iters 671088640 cells 10.00000000 GBs OMP [/opt/rocm-4.0.1/llvm/bin/clang++]
  0.000 (0.000s): Version 1.1 (3/22/21)
  0.000 (0.000s): Allocating and initializing CPU arrays
  0.000 (0.000s): Starting omp data map of CPU arrays to GPU
  5.240 (5.240s): Finished omp data map of CPU arrays to GPU
  5.240 (0.000s): Launching warmup iteration (not included in timings)
  5.283 (0.043s): Starting iteration      1
  5.691 (0.408s): Starting iteration     11
  6.099 (0.409s): Starting iteration     21
  6.509 (0.410s): Starting iteration     31
  6.918 (0.409s): Starting iteration     41
  7.329 (0.411s): Starting iteration     51
  7.741 (0.411s): Starting iteration     61
  8.153 (0.412s): Starting iteration     71
  8.565 (0.412s): Starting iteration     81
  8.977 (0.412s): Starting iteration     91
  9.389 (0.412s): STATS Rush Larsen 100 iters 4.1062 sec 41062.27 us/iter 5.240 sec datatrans OMP [/opt/rocm-4.0.1/llvm/bin/clang++]
  9.389 (0.000s): PASSED GPU Data sanity check m_gate[0]=0.996321172062538 OMP [/opt/rocm-4.0.1/llvm/bin/clang++]
  9.439 (0.049s): DONE Freed CPU and GPU memory OMP [/opt/rocm-4.0.1/llvm/bin/clang++]


