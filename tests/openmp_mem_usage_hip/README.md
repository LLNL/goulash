# Goulash openmp_mem_usage_hip

This is an important LLNL GPU use case reproducer, openmp_mem_usage_hip, 
that many of our simulation codes rely on working well.  It allocates a
command-line specified amount of GPU memory via OpenMP map and then
uses openMP omp_soft_pause and omp_hard_pause to attempt to free that memory.
It then uses and frees all GPU memory via hipMalloc and hipFree several times,
reporting how much memory it could allocate.  This simulates a package using
OpenMP GPU code to use the GPU and then other packages using HIP to use the
GPU in a single simulation cycle. The first parameter specifies how many
times to iterate (i.e., how many simulation cycles to emulate).
Our apps want to be able to do this thousands of times in a 24 hour run.
Our acceptance tests generally run at least for 100 iterations to find
leaks and other undesirable behavior that have been detected in the past.

This test is one of the building blocks we are using for interoperability
tests (building with multiple compilers). An interop version will be 
added soon to goulash.

Note: At the time of this initial release, the compilers were too new and 
evolving quickly to be in cmake, autotools, etc.    We provided some build
scripts on various platforms we tested on to get you started but you will
likely have to figure out your build line manually for now.   Allowing
build systems to figure out how to build these complex cases is one of 
the goals of goulash.

Here is a sample usage info and build state message from one of the sample
builds (run executable with -h or no arguments):

Usage: ./openmp_mem_usage_hip_cce_amd  Iterations  Kernel_GBs_used

Measure fixed overheads: ./openmp_mem_usage_hip_cce_amd   1 .0000001
  Measure all overheads: ./openmp_mem_usage_hip_cce_amd   1 31
 Runtime stability test: ./openmp_mem_usage_hip_cce_amd 100 31

Version 1.1 (12/18/20)

Measures GPU memory available with OpenMP offloading before/after kernel launch,
data map, and omp_soft_pause and omp_hard_pause (Compiled without -DNO_OPENMP)

Uses hipHostMalloc to allocate pinned bufs   (Compiled without -DUSE_CALLOC)
All pinned hipHostMalloc bufs will be freed  (Compiled without -DLEAK_HOST_MEM)

All GPUs will be tested (Compiled without -DSINGLE_GPU)


