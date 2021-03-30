# Goulash rush_larsen_hip

This test (no MPI, hip-kernels using GPU) is one of the building blocks we are using
for interoperability tests (building with multiple compilers in the same
application). An interop version will be added soon to goulash.
Building and verifying the goulash components separately is recommmended.

Note: With this initial release, we provided a trivial Makefile using 
ROCM to get you started but you will likely have to figure
out your build line manually for now.   Providing relevant testcases 
for build systems in order to figure out how to build these complex 
cases is one of the goals of goulash.

Here is a sample usage info (run executable with -h or no arguments):

Usage: ./rush_larsen_hip  Iterations  Kernel_GBs_used

Measure launch overhead: ./rush_larsen_hip 100000 .00000001
Measure GPU performance: ./rush_larsen_hip    100 10

Version 1.1 (3/22/21)

Here is a sample benchmark output from a single AMD MI60 GPU (rocm-4.0.1):
./rush_larsen_hip    100 10
  0.000 (0.000s): START Rush Larsen 100 iters 671088640 cells 10.00000000 GBs HIP [/opt/rocm-4.0.1/bin/hipcc]
  0.000 (0.000s): Version 1.1 (3/22/21)
  0.000 (0.000s): Allocating and initializing CPU arrays
  0.000 (0.000s): Allocating GPU arrays
  0.001 (0.001s): Starting hipMemcpy CPU arrays to GPU arrays
  4.234 (4.233s): Finished hipMemcpy CPU arrays to GPU arrays
  4.234 (0.000s): Launching warmup iteration (not included in timings)
  4.261 (0.028s): Starting iteration      1
  4.492 (0.231s): Starting iteration     11
  4.727 (0.234s): Starting iteration     21
  4.964 (0.237s): Starting iteration     31
  5.198 (0.234s): Starting iteration     41
  5.431 (0.233s): Starting iteration     51
  5.665 (0.234s): Starting iteration     61
  5.898 (0.233s): Starting iteration     71
  6.138 (0.240s): Starting iteration     81
  6.372 (0.233s): Starting iteration     91
  6.605 (0.234s): STATS Rush Larsen 100 iters 2.3440 sec 23440.27 us/iter 4.233 sec datatrans HIP [/opt/rocm-4.0.1/bin/hipcc]
  6.606 (0.000s): PASSED GPU Data sanity check m_gate[0]=0.996321172062538 HIP [/opt/rocm-4.0.1/bin/hipcc]
  7.164 (0.559s): DONE Freed CPU and GPU memory HIP [/opt/rocm-4.0.1/bin/hipcc]

