# Goulash rush_larsen_cpu

This test (no MPI, no GPU) is one of the building blocks we are using
for interoperability tests (building with multiple compilers in the same
application). An interop version will be added soon to goulash.
Building and verifying the goulash components separately is recommmended.

Note: With this initial release, we provided a trivial Makefile using 
ROCM to get you started but you will likely have to figure
out your build line manually for now.   Providing relevant testcases 
for build systems in order to figure out how to build these complex 
cases is one of the goals of goulash.

Here is a sample usage info (run executable with -h or no arguments):

Usage: ./rush_larsen_cpu  Iterations  Kernel_GBs_used

Measure serial launch overhead:   env OMP_NUM_THREADS=1 ./rush_larsen_cpu 100000 .00000001
Measure thread launch overhead:   ./rush_larsen_cpu 100000 .00000001
Measure CPU threaded performance: ./rush_larsen_cpu    100 10

Version 1.1 (3/22/21)

Here is a sample benchmark output from an 36-core x86 node with hyperthreading:
./rush_larsen_cpu    100 10
  0.000 (0.000s): START Rush Larsen 100 iters 671088640 cells 10.00000000 GBs CPU [icc-19.0.4]
  0.000 (0.000s): Version 1.1 (3/22/21)
  0.049 (0.049s): Initial OpenMP Map 0-17,19-34,37-41,44,47,51,54,56,58,62,71 Threads 47 rztopaz2
  0.049 (0.000s): Allocating and initializing CPU arrays
  0.049 (0.000s): Launching warmup iteration (not included in timings)
  4.773 (4.724s): Post-warmup OpenMP Map 0-2,4-5,7-10,12-19,21-22,24-27,29-71 Threads 66 rztopaz2
  4.773 (0.000s): Starting iteration      1
 11.503 (6.730s): Starting iteration     11
 18.144 (6.641s): Starting iteration     21
 24.808 (6.664s): Starting iteration     31
 31.393 (6.585s): Starting iteration     41
 38.032 (6.640s): Starting iteration     51
 44.629 (6.597s): Starting iteration     61
 51.192 (6.562s): Starting iteration     71
 57.778 (6.586s): Starting iteration     81
 64.371 (6.594s): Starting iteration     91
 71.032 (6.661s): STATS Rush Larsen 100 iters 66.2593 sec 662593.19 us/iter CPU [icc-19.0.4]
 71.032 (0.000s): PASSED Data sanity check m_gate[0]=0.996321172062538 CPU [icc-19.0.4]
 71.042 (0.009s): Final OpenMP Map 0-71 Threads 72 rztopaz2
 71.464 (0.422s): DONE Freed memory CPU [icc-19.0.4]


