! Copyright (c) 2019-21, Lawrence Livermore National Security, LLC. and other
! Goulash project contributors LLNL-CODE-795383, All rights reserved.
! For details about use and distribution, please read LICENSE and NOTICE from
! the Goulash project repository: http://github.com/llnl/goulash
! SPDX-License-Identifier: BSD-3-Clause
!
! Designed to allow direct performance comparisons between
! naively written HIP/CUDA and OpenMP GPU offloading schemes
! in a variety of coding styles and languages of a
! parameterized embarrassingly parallel Rush Larsen kernel.
! Also allows testing build systems (including SPACK) handling
! of complicated build situations that LLNL cares about.
!
! Designed to create several single file test variants
! where no -D options required to select the variant
! and no include files are needed.
! Almost all code in this file is identical 
! between variants (this is intentional).
! MPI support is ifdefed out for non-MPI variants.
!
! The key computational kernel can be located by
! searching for: KERNEL
!
! Designed to create trivial variants of the same
! test to be compiled with different compilers for
! the interoperability tests.   This is why most
! functions are static and the main kernel test
! is called RUSH_LARSEN_VARIANT (preprocessor
! defines this and VARIANT_DESC based on
! variant configuration).
!
! The naming convention of the variant copies is
! intended to indicate variant abilities:
! _cpu_serial      - single threaded, no OpenMP, on CPU
! _cpu_omp         - use OpenMP to spawn threads on CPU
! _gpu_omp         - uses OpenMP to offload to GPU
! _gpu_hip         - uses HIP to offload to AMD or Nvidia GPU
! _gpu_lambda_hip  - RAJA-like lambda HIP variant
! _gpu_cuda        - uses CUDA to offload to Nvidia GPU
! _gpu_lambda_cuda - RAJA-like lambda CUDA variant
! *_mpi            - uses and exercises MPI e.g. _gpu_omp_mpi
! *_fort           - Fortran version e.g. _gpu_omp_mpi_fort
!
! For the interop tests, there is an additional suffix
! to indicate different copies of the same configuration
! that are intended to be compiled by different compilers:
! _compiler1  - E.g., rush_larsen_gpu_omp_compiler1.cc
!
! VARIANT_DESC set by preprocessor directives to
! the configuration of this file.
!
! Recommended that a -DCOMPILERID be set to the compiler used to compile each file:
!
! /opt/rocm-4.0.1/llvm/bin/clang++  -o rush_larsen_cpu_omp -O3 -g "-DCOMPILERID=rocm-4.0.1" -fopenmp rush_larsen_cpu_omp.cc
!
! Run with no arguments for suggested arguments, for example:
!   Usage: ./rush_larsen_cpu_omp  Iterations  Kernel_GBs_used
!
!     Measure serial launch overhead:  env OMP_NUM_THREADS=1 ./rush_larsen_cpu_omp 100000 .00000001
!     Measure launch overhead:         ./rush_larsen_cpu_omp 100000 .00000001
!     Measure kernel performance:      ./rush_larsen_cpu_omp    100 10
!
!
! The Goulash project conceived of and designed by David Richards,
! Tom Scogland, and John Gyllenhaal at LLNL Oct 2019.
! Please contact John Gyllenhaal (gyllenhaal1@llnl.gov) with questions.
!
! Rush Larsen core CUDA/OpenMP kernels written by Rob Blake (LLNL) Sept 2016.
! The goulash Rush Larsen tests add benchmarking infrastructure
! around this incredibly useful compact GPU test kernel.   Thank you Rob!
!
! Inline performance measurements added (nvprof not needed)
! by John Gyllenhaal at LLNL 11/10/20.
!
! Command line argument handling, performance difference printing in
! form easily found with grep, OpenMP thread mapping and initial data
! sanity checks on just the first array element calculated by kernel
! by John Gyllenhaal at LLNL 03/22/21
!
! Pulled code from print_openmp_mapping.c by John Gyllenhaal at
! LLNL written June 2020 which was based on mpibind tests
! (https://github.com/LLNL/mpibind) by Edgar Leon at LLNL
!
! RAJA-perf-suite-like (https://github.com/LLNL/RAJAPerf)
! C++ lambda versions created by Jason Burmark at LLNL 06/16/21
!
! C-like Fortran ports (by hand) of C++ version of
! rush larsen variants by John Gyllenhaal at LLNL 06/28/21.
!
! MPI stat aggregation and MPI exercisers written
! by John Gyllenhaal at LLNL based on previous user issue reproducers.
! Pulled into rush larsen tests on 07/02/21 by John Gyllenhaal at LLNL.
!
! Enhanced data checks of all kernel generated array data, including
! across MPI ranks by John Gyllenhaal at LLNL 07/03/21
!
! Interop versions create by John Gyllenhaal at LLNL 07/14/21
! to test mixing all the Rush Larsen tests with multiple GPU compilers
! all in one final executable.
!
! Initial test generator from template files, including Makefiles
! created by John Gyllenhaal at LLNL 07/21/21 for V2.0RC1
!
! V2.0 RC1 07/21/21 Added MPI support, interop version, enhanced data checks.
! V1.2 06/28/21 Fortran and C++ lambda versions added, consistent use of long type
! V1.1 03/22/21 command line args, perf diffs, maps, checks return codes and answer
! V1.0 11/10/20 initial release, hard coded inputs, no error checking

! Allow version to be printed in output 
#define VERSION_STRING "Version 2.0 RC1 (7/21/21)"

! Set -DGOULASH_UNLOCK to be able to overwrite hard coded variant settings
! on compile line.
!
! By default, each variant of this file is locked to a particular configuration,
! ignoring compile line options.   This is the recommended usage mode!
!
! These additional locked variant suboptions can be used to
! compile out some optional functionality of the tests:
!
!   NO_THREAD_MAP     - Do not print threadmap for TARGET_CPU_OMP
!   NO_MPI_EXERCISERS - Do not exercise MPI with MPI variants
!   NO_MAIN           - Do not use main driver, used for interop tests
!   APPEND_COMPILER1  - Append _compiler1 to function name for interop tests
!
! Unlocking designed to allow regression testing on all supported variants
! from one source file before copying and hard coding these settings
! for each variant when pushing out a new Goulash version.
!
! NOTE: Some Fortran preprocessors (especially Cray's) cannot handle 
!       concatination and stringify, so we hardcoded the base and
!       interop _compiler1 variations without those cpp tricks.

! Unless unlocked, define variant for this file.
#ifndef GOULASH_UNLOCK
! MUST BE EXACTLY TWO SPACES BETWEEN #undef AND VARIABLE 
! THIS IS REQUIRED FOR THE TEMPLATE INSTANTIATION 

! Exactly one TARGET must be defined to determine variant used
! These fortran tests only exercise OpenMP targets right now
#undef  TARGET_CPU_SERIAL
#undef  TARGET_CPU_OMP
#undef  TARGET_GPU_OMP

! Any target may use MPI to aggregate results and exercise MPI
#undef  USE_MPI

! Additional variant suboptions
! NOTE: Cannot have comments on same lines as #undef due to generation logic!
!
! Do not print threadmap when have TARGET_CPU_OMP
#undef  NO_THREAD_MAP

! Do not exercise MPI with MPI variants 
#undef  NO_MPI_EXERCISERS

! Do not use main driver, used for interop tests
#undef  NO_MAIN

! Used by interop tests to append _compiler1 to routine's name
#undef  APPEND_COMPILER1

! end if GOULASH_UNLOCK
#endif 

!
! Make sure exactly one TARGET is defined
! Generate preprocessing error otherwise.
!
#if !((defined(TARGET_CPU_SERIAL) + defined(TARGET_CPU_OMP) + defined(TARGET_GPU_OMP)) == 1)
! Do not have exactly one target
! Indicate if zero targets or multiple targerts is the problem
!
#if ((defined(TARGET_CPU_SERIAL) + defined(TARGET_CPU_OMP) + defined(TARGET_GPU_OMP)) == 0)
#error Zero TARGETs defined.   Exactly one TARGET must be defined to compile Goulash rush_larsen Fortran varations
#else
#error Multiple TARGETs defined.   Exactly one TARGET must be defined to compile Goulash rush_larsen Fortran variations
#endif

! With some preprocessors these tests will show what is multiply defined
#ifdef TARGET_CPU_SERIAL
#error Multiple targets specified: TARGET_CPU_SERIAL
#endif
#ifdef TARGET_CPU_OMP
#error Multiple targets specified: TARGET_CPU_OMP
#endif
#ifdef TARGET_GPU_OMP
#error Multiple targets specified: TARGET_GPU_OMP
#endif
#endif

! Create unique VARIANT_TAG based on file #def and #undef
! settings that is used to create rush_larsen function
! call name and to annotate key lines of output

! For interop testcase, need to append _compiler1 to function name
! but some fortran compilers (e.g. Cray) do not allow concatenating 
! values so hardcode support for _compiler1
#if defined(APPEND_COMPILER1)

#if defined(TARGET_CPU_SERIAL) && !defined(USE_MPI) 
#define VARIANT_TAG "cpu_serial_fort_compiler1"
#define RUSH_LARSEN_VARIANT rush_larsen_cpu_serial_fort_compiler1

#elif defined(TARGET_CPU_SERIAL) && defined(USE_MPI)
#define VARIANT_TAG "cpu_serial_mpi_fort_compiler1"
#define RUSH_LARSEN_VARIANT rush_larsen_cpu_serial_mpi_fort_compiler1

#elif defined(TARGET_CPU_OMP) && !defined(USE_MPI) 
#define VARIANT_TAG "cpu_omp_fort_compiler1"
#define RUSH_LARSEN_VARIANT rush_larsen_cpu_omp_fort_compiler1

#elif defined(TARGET_CPU_OMP) && defined(USE_MPI)
#define VARIANT_TAG "cpu_omp_mpi_fort_compiler1"
#define RUSH_LARSEN_VARIANT rush_larsen_cpu_omp_mpi_fort_compiler1

#elif defined(TARGET_GPU_OMP) && !defined(USE_MPI)
#define VARIANT_TAG "gpu_omp_fort_compiler1"
#define RUSH_LARSEN_VARIANT rush_larsen_gpu_omp_fort_compiler1

#elif defined(TARGET_GPU_OMP) && defined(USE_MPI)
#define VARIANT_TAG "gpu_omp_mpi_fort_compiler1"
#define RUSH_LARSEN_VARIANT rush_larsen_gpu_omp_mpi_fort_compiler1

#else
#error unexpected rush_larsen #define configuration with _compiler1
#endif

! Normal case, don't append _compiler1
#else

#if defined(TARGET_CPU_SERIAL) && !defined(USE_MPI)
#define VARIANT_TAG "cpu_serial_fort"
#define RUSH_LARSEN_VARIANT rush_larsen_cpu_serial_fort

#elif defined(TARGET_CPU_SERIAL) && defined(USE_MPI)
#define VARIANT_TAG "cpu_serial_mpi_fort"
#define RUSH_LARSEN_VARIANT rush_larsen_cpu_serial_mpi_fort

#elif defined(TARGET_CPU_OMP) && !defined(USE_MPI)
#define VARIANT_TAG "cpu_omp_fort"
#define RUSH_LARSEN_VARIANT rush_larsen_cpu_omp_fort

#elif defined(TARGET_CPU_OMP) && defined(USE_MPI)
#define VARIANT_TAG "cpu_omp_mpi_fort"
#define RUSH_LARSEN_VARIANT rush_larsen_cpu_omp_mpi_fort

#elif defined(TARGET_GPU_OMP) && !defined(USE_MPI) 
#define VARIANT_TAG "gpu_omp_fort"
#define RUSH_LARSEN_VARIANT rush_larsen_gpu_omp_fort

#elif defined(TARGET_GPU_OMP) && defined(USE_MPI)
#define VARIANT_TAG "gpu_omp_mpi_fort"
#define RUSH_LARSEN_VARIANT rush_larsen_gpu_omp_mpi_fort

#else
#error unexpected rush_larsen #def configuration 
#endif

#endif

! Sets up and runs the doRushLarsen kernel 'iterations' times, 
! allocating CPU arrays and perhaps GPU arrays to consume 
! kernel_mem_used GBs of memory.
!
! This polynomial is a fit to the dynamics of a small part of a cardiac
! myocyte, specifically the fast sodium m-gate described here:
! https://www.ncbi.nlm.nih.gov/pubmed/16565318
!
! Does exactly the same work on every cell.   Can scale from one cell
! to filling entire memory.   Does use cell's value as input
! to calculations.
!
! Returns number of data check failures, returns 0 if all data checks out.
function RUSH_LARSEN_VARIANT(iterations_, kernel_mem_used_)

  ! Only include mpi for variants that need it
#ifdef USE_MPI
  use mpi
#endif

  ! Only include OpenMP for variants that need it
#if defined(TARGET_CPU_OMP) || defined(TARGET_GPU_OMP)
  use omp_lib
#endif
  ! Get mappings to stdout, etc. so can flush output
  use, intrinsic :: iso_fortran_env, only : stdin=>input_unit, &
       &                                    stdout=>output_unit, &
       &                                    stderr=>error_unit

  ! Catch misspelled variables 
  implicit none

  ! Declare arguments
  integer :: RUSH_LARSEN_VARIANT
  integer(8),  intent(IN) :: iterations_
  integer(8) :: iterations
  real(8), intent(IN) :: kernel_mem_used_
  real(8) :: kernel_mem_used

  ! Declare local variables
  ! NOTE: All subroutines and functions called by this routine
  !       can access these variables!
  !       Used for variant_desc, kernel_mem_used_str, timestamp, sec_str, us_str, ierr
  character(1024) :: variant_desc
  character(50) :: timestamp
  character(50) :: kernel_mem_used_str
  character(50) :: sec_str, us_str, transfer_str
  real(8) :: kernel_starttime, kernel_endtime, kernel_runtime, base_time, last_timestamp, cur_secs
  real(8) :: transfer_starttime, transfer_endtime, transfer_runtime
  integer(8) :: nCells, status_point
  integer :: rank = 0 ! Rank will be 0 for the no MPI case 
  integer :: ierr
  integer(8) :: fail_count = 0
  real(8) :: sum1, sum2, x, mhu, tauR
  integer(8) :: itime, ii
  integer(4) :: j, k
  real(8), allocatable :: m_gate(:), Vm(:)
  integer(4), parameter :: Mhu_l = 10
  integer(4), parameter :: Mhu_m = 5
  integer(4), parameter :: Tau_m = 18
  ! Must use 'd' in every constant in order to get full real*8 values and matching results
  real(8) :: Mhu_a(0:14) = (/&
       &  9.9632117206253790d-01,  4.0825738726469545d-02,  6.3401613233199589d-04,&
       &  4.4158436861700431d-06,  1.1622058324043520d-08,  1.0000000000000000d+00,&
       &  4.0568375699663400d-02,  6.4216825832642788d-04,  4.2661664422410096d-06,&
       &  1.3559930396321903d-08, -1.3573468728873069d-11, -4.2594802366702580d-13,&
       &  7.6779952208246166d-15,  1.4260675804433780d-16, -2.6656212072499249d-18/)
  ! Must use 'd' in every constant in order to get full real*8 values and matching results
  real(8) :: Tau_a(0:18) = (/&
       &  1.7765862602413648d+01*0.02d+00,  5.0010202770602419d-02*0.02d+00, -7.8002064070783474d-04*0.02d+00,&
       & -6.9399661775931530d-05*0.02d+00,  1.6936588308244311d-06*0.02d+00,  5.4629017090963798d-07*0.02d+00,&
       & -1.3805420990037933d-08*0.02d+00, -8.0678945216155694d-10*0.02d+00,  1.6209833004622630d-11*0.02d+00,&
       &  6.5130101230170358d-13*0.02d+00, -6.9931705949674988d-15*0.02d+00, -3.1161210504114690d-16*0.02d+00,&
       &  5.0166191902609083d-19*0.02d+00,  7.8608831661430381d-20*0.02d+00,  4.3936315597226053d-22*0.02d+00,&
       & -7.0535966258003289d-24*0.02d+00, -9.0473475495087118d-26*0.02d+00, -2.9878427692323621d-28*0.02d+00,&
       &  1.0000000000000000d+00/)
#ifdef USE_MPI
  ! ALL contains routines can access rank, numtasks, etc.
  ! so initialize them once per call to main kernel routine
  integer:: numtasks
  real(8):: dnumtasks
#endif

  ! Allow compiler to be passed in at compile time
  ! Must pass in quoted string on command line, i.e., '-DCOMPILERID="CCE"' 
#ifdef COMPILERID
  write (variant_desc, 20) VARIANT_TAG, ' [', COMPILERID , ']'
20 format(a, a,a,a)
#else
  variant_desc=VARIANT_TAG
#endif

#ifdef USE_MPI
  ! Get actual MPI rank if using MPI practically all
  ! functions and subroutines in this file
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

  ! Get number of tasks for all 'contains' routines
  call MPI_Comm_size(MPI_COMM_WORLD, numtasks, ierr)

  ! Use double version of numtasks in calculations
  dnumtasks = numtasks

  ! Sync all the MPI ranks before starting
  call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif

  ! To make interop performance easier to compare,
  ! start this file's timers over every time called.
  !
  ! Reset this file's secs_elapsed() counter to 0 
  cur_secs = get_raw_secs()
  base_time = get_base_time(cur_secs)

  ! Synchronize printf timestamps across MPI ranks
  last_timestamp = get_last_timestamp(secs_elapsed())

  if (rank == 0) then
     ! Print separator before and after output with function name
     call get_timestamp_string(timestamp)
     print '(a,"--------------- Begin rush_larsen_",a," (timer zeroed) ---------------")', &
          & trim(timestamp), trim(VARIANT_DESC)
     flush(stdout)
  endif



  ! For print niceness, make .00000001 lower bound on GB memory
  if (kernel_mem_used_ < .00000001) then
     kernel_mem_used = .00000001
  else
     kernel_mem_used = kernel_mem_used_
  end if

  ! Calculate nCells from target memory target
  nCells =  ((kernel_mem_used * 1024.0 * 1024.0 * 1024.0) / (8 * 2))

  ! Must have at least 1 cell 
  if (nCells < 1) then
     nCells = 1
  end if

  ! Must have at least 1 iteration 
  if (iterations_ < 1) then
     iterations=1
  else
     iterations=iterations_
  end if

  ! Give status every 10% of iterations 
  status_point=iterations/10
  ! Must be at least 1 to make mod work
  if (status_point < 1) then
     status_point = 1
  end if

  ! Print what we are running
  ! Convert kernel_mem_used to left justified string with leading 0
  ! This str is used in other subroutines and functions
  write (kernel_mem_used_str, 50) kernel_mem_used
50 format(F16.8)
  kernel_mem_used_str=adjustl(kernel_mem_used_str)
  ! This kernel_mem_used_str used in several other messages as is

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," START Rush Larsen ",i0," iters ",i0," cells ",a," GBs ",a)',&
          & trim(timestamp),iterations, nCells, trim(kernel_mem_used_str), trim(variant_desc)

     call get_timestamp_string(timestamp)
     print '(a," ",a)', trim(timestamp), trim(VERSION_STRING)
     flush(stdout)
  endif

#ifdef TARGET_GPU_OMP
  ! If using OpenMP offloading, make sure GPU works before doing test
  call verify_gpu_openmp(0)
#endif

#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP)
  ! Print OpenMP thread mapping, syncs and aggregates MPI ranks (if MPI mode)
  call print_thread_stats ("Initial OpenMP")
#endif

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," ",a)', trim(timestamp), "Allocating and initializing kernel arrays"
     flush(stdout)
  endif

  ! Porting from C, so make all arrays start at index 0 to make port easier
  allocate(m_gate(0:nCells-1))
  m_gate=0.0

  ! Porting from C, so make all arrays start at index 0 to make port easier
  allocate(Vm(0:nCells-1))
  Vm=0.0

  ! No data transfer time if not using GPU 
  transfer_starttime=0.0
  transfer_endtime=0.0

#if defined(TARGET_GPU_OMP)
#ifdef USE_MPI
  ! Sync all the MPI ranks before starting 
  call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif
  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," Starting omp data map of CPU arrays to GPU")', trim(timestamp)
     flush(stdout)
  end if
  transfer_starttime=secs_elapsed()
  !$omp target enter data map(to: m_gate(0:nCells-1))
  !$omp target enter data map(to: Vm(0:nCells-1))
  !$omp target enter data map(to: Mhu_a(0:14))
  !$omp target enter data map(to: Tau_a(0:18))
  transfer_endtime=secs_elapsed()
#ifdef USE_MPI
  ! Sync all the MPI ranks before printing message 
  call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif
  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," Finished omp data map of CPU arrays to GPU")', trim(timestamp)
     flush(stdout)
  end if
  ! end if defined(TARGET_GPU_OMP)
#endif

  transfer_runtime=transfer_endtime-transfer_starttime

  ! Do the iterations asked for plus 1 for warmup
  do itime=0,iterations
     ! Print warmup message for 0th iteration
     if (itime == 0) then
        if (rank == 0) then
           call get_timestamp_string(timestamp)
           print '(a,a)', trim(timestamp), " Launching warmup iteration (not included in timings)"
           flush(stdout)
        endif
        ! Print status every 10% of iterations
     else if (modulo((itime-1), status_point) == 0) then
        if (itime == 1) then
#ifdef USE_MPI
           if (rank == 0) then
              call get_timestamp_string(timestamp)
              print '(a,a)', trim(timestamp), " Waiting for all MPI ranks to complete warmup"
              flush(stdout)
           endif
           ! Wait for all MPI ranks to complete
           call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif
#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP)
           ! Print OpenMP thread mapping, syncs and aggregates MPI ranks (if MPI mode)
           call print_thread_stats ("Post-warmup OpenMP")
#endif

#if defined(USE_MPI) && !defined(NO_MPI_EXERCISERS)
           ! Do some short MPI exercisers that have exposed issues before (if in MPI mode)
           call mpi_exerciser_driver("Post-warmup")

#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP)
           ! Print OpenMP thread mapping after MPI to see if impacted
           call print_thread_stats ("Post-MPI OpenMP")
#endif
           ! end if defined(USE_MPI) && !defined(NO_MPI_EXERCISERS)
#endif
        endif

        if (rank == 0) then
           call get_timestamp_string(timestamp)
           if (itime == 1) then
              print '(a," Starting kernel timings for Rush Larsen ",i0," ",a)',&
                   & trim(timestamp),iterations, trim(kernel_mem_used_str)
           endif
           print '(a,a,i6)', trim(timestamp), " Starting iteration ", itime
           flush(stdout)
        endif
     end if

     ! Start timer after warm-up iteration 0
     if (itime == 1) then
        kernel_starttime = secs_elapsed()
     end if

     !
     ! RUSH LARSEN KERNEL BEING TIMED START
     !
#if defined(TARGET_GPU_OMP)
     ! Target GPU with OpenMP, data already mapped to GPU 
     !$omp target teams distribute parallel do simd private(ii,x,sum1,j,sum2,k,mhu,tauR)

#elif defined(TARGET_CPU_OMP)
     ! Target CPU with OpenMP parallelism
     !$omp parallel do private(ii,x,sum1,j,sum2,k, mhu,tauR)

#elif defined(TARGET_CPU_SERIAL)
     ! Serial CPU execution, basic run 
#endif
     do ii=0,nCells-1
        x = Vm(ii)
        sum1 = 0.0
        do j = Mhu_m-1, 0, -1
           sum1 = Mhu_a(j) + x*sum1
        end do
        sum2 = 0.0
        k = Mhu_m + Mhu_l - 1
        do j = k, Mhu_m, -1
           sum2 = Mhu_a(j) + x * sum2
        end do
        mhu = sum1/sum2

        sum1 = 0.0
        do j = Tau_m-1, 0, -1
           sum1 = Tau_a(j) + x*sum1
        end do
        tauR = sum1

        m_gate(ii) = m_gate(ii) + (mhu - m_gate(ii))*(1-exp(-tauR))
     end do
#if defined(TARGET_GPU_OMP)
     ! End Target GPU with OpenMP, data already mapped to GPU 
     !$omp end target teams distribute parallel do simd 

#elif defined(TARGET_CPU_OMP)
     ! End Target CPU with OpenMP parallelism
     !$omp end parallel do
#endif
     !
     ! RUSH LARSEN KERNEL BEING TIMED END
     ! 
  end do

  ! Get time after all iterations
  kernel_endtime = secs_elapsed ()

  ! Calculate kernel runtime
  kernel_runtime = kernel_endtime-kernel_starttime

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a,a,i0,a,a)',&
          & trim(timestamp)," Finished kernel timings for Rush Larsen ", iterations, " ", trim(kernel_mem_used_str) 
     flush(stdout)

  endif

#ifdef USE_MPI
  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a,a)', trim(timestamp), " Waiting for all MPI ranks to complete calculations"
     flush(stdout)
  endif
  ! Wait for all MPI ranks to complete
  call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif

#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP)
  ! Print OpenMP thread mapping, syncs and aggregates MPI ranks (if MPI mode)
  call print_thread_stats ("Final OpenMP")
#endif

  ! Print kernel runtime stats, syncs and aggregates MPI rank (if MPI mode)
  call print_runtime_stats(iterations, kernel_mem_used, kernel_runtime, transfer_runtime)

#ifdef TARGET_GPU_OMP
  ! Transfer GPU m_gate kernel memory to CPU kernel memory for data checks 
  !$omp target update from (m_gate(0:nCells-1))
#endif

  ! Do sanity and consistency checks on all of m_gate. Including cross-rank if in MPI mode.
  ! Prints PASS or FAIL based on data check results
  ! Returns fail count so can be returned to caller.
  fail_count = data_check (m_gate, iterations, kernel_mem_used, nCells)

#ifdef TARGET_GPU_OMP
  ! Free kernel GPU memory
  !$omp target exit data map(delete: m_gate(0:nCells-1))
  !$omp target exit data map(delete: Vm(0:nCells-1))
  !$omp target exit data map(delete: Mhu_a(0:14))
  !$omp target exit data map(delete: Tau_a(0:18))
#endif

  deallocate(Vm)
  deallocate(m_gate)

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," ",a,a)', trim(timestamp), "DONE Freed memory ", trim(variant_desc)
     flush(stdout)
     ! Print separator before and after output with function name
     call get_timestamp_string(timestamp)
     print '(a,"--------------- End rush_larsen_",a," ---------------")', &
          & trim(timestamp), trim(VARIANT_DESC)
     flush(stdout)
  endif

  ! Return number of data check failures
  RUSH_LARSEN_VARIANT = fail_count

contains

  ! Ends program either with MPI_Abort or STOP 1
  subroutine die()
#ifdef USE_MPI
    integer ierr
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
#else
    stop 1
#endif
  end subroutine die

  ! Get raw time in seconds as double (a large number).
  function get_raw_secs()
    ! Catch misspelled variables
    implicit none
    real(8) :: get_raw_secs
    integer(8) :: count, count_rate, count_max
    real(8) :: dcount, dcount_rate

    ! Get wall-clock time
    call system_clock(count, count_rate, count_max)
    dcount = count
    dcount_rate = count_rate
    ! Convert values to double (in seconds ) (a large number)
    get_raw_secs = dcount/dcount_rate
  end function get_raw_secs

  ! Returns base time.  If new_time > 0,
  ! sets base_time to new_time before returning.
  ! Using this as access method to static variable
  ! in a way I can trivially emulate in fortran.
  !
  ! Note: Lock shouldn't be needed, since even if multiple
  !       threads initialize this, it will be to basically
  !       the same value.
  !
  function get_base_time(new_time) 
    ! Catch misspelled variables
    implicit none
    real(8), intent(IN):: new_time
    real(8):: get_base_time
    real(8), save :: base_time = -1.0

    !If passed value > 0
    if (new_time > 0.0) then
       base_time = new_time
    end if

    get_base_time = base_time
  end function get_base_time


  ! Returns time in seconds (double) since the first call to secs_elapsed
  ! (i.e., the first call returns 0.0).
  function secs_elapsed ()
    ! Catch misspelled variables 
    implicit none
    real(8) :: secs_elapsed
    real(8) :: new_time, base_time

    ! Get current raw time (a big number)
    new_time = get_raw_secs()

    base_time = get_base_time(-1.0_8)

    ! If base time not set (negative), set to current time (pass in positive secs)
    if (base_time < 0.0) then
       base_time=get_base_time(new_time)
    end if

    ! Returned offset from first time called
    secs_elapsed = new_time - base_time
  end function secs_elapsed

  function get_last_timestamp(new_time) 
    ! Catch misspelled variables
    implicit none
    real(8), intent(IN):: new_time
    real(8):: get_last_timestamp
    real(8), save :: last_timestamp = -1.0

    !If passed value > 0
    if (new_time >= 0.0) then
       last_timestamp = new_time
    end if

    get_last_timestamp = last_timestamp
  end function get_last_timestamp


  ! Cannot wrap print in fortran so create utility function
  ! for creating timestamp prefix with diff from last timestamp.
  ! Generate timestamp string of this form:
  !    0.095 (0.000s): 
  subroutine get_timestamp_string (timestamp_string)
    ! Only include mpi for variants that need it
#ifdef USE_MPI
    use mpi
#endif
    ! Catch misspelled variables 
    implicit none
    character(len=*), intent(OUT) :: timestamp_string
    real(8) :: last_timestamp
    real(8) :: sec, diff
    integer :: rank = -1

#ifdef USE_MPI
    integer :: ierr
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
#endif
    ! Get wall-clock time since first call to secs_elapsed
    sec = secs_elapsed ()

    ! Query last timestamp, set first time if needed
    last_timestamp = get_last_timestamp(-1.0_8)
    if (last_timestamp < 0.0) then
       last_timestamp=get_last_timestamp(sec)
    end if

    diff = sec - last_timestamp

    ! Set new last timestamp
    last_timestamp=get_last_timestamp(sec)


    ! No MPI case
    if (rank < 0) then
       ! Write out timestamp and diff seconds to buffer
       write (timestamp_string, 10) sec, ' (', diff, 's): '
10     format(f7.3,a,f5.3,a)

       ! MPI case, add rank
    else
       ! Write out timestamp and diff seconds to buffer
       write (timestamp_string, 11) rank, ": ", sec, ' (', diff, 's): '
11     format(i3,a,f7.3,a,f5.3,a)
    endif
  end subroutine get_timestamp_string

#if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP)
  ! Spawns OpenMP threads and prints cpu mappings.
  ! Send rank -1 if not in MPI program (or don't want rank printed).
  function print_openmp_mapping (variant_desc, rank)
    ! Get mappings to stdout, etc. so can flush output
    use, intrinsic :: iso_fortran_env, only : stdin=>input_unit, &
         &                                         stdout=>output_unit, &
         &                                         stderr=>error_unit
    use omp_lib
    use iso_c_binding

    ! Catch misspelled variables 
    implicit none

    ! Declare arguments
    integer(4),  intent(IN) :: rank
    character(len=*), intent(in) :: variant_desc
    integer:: print_openmp_mapping

    interface
       integer function sched_getcpu() bind(C, name="sched_getcpu")
       end function sched_getcpu
    end interface


    ! Declare local variables
    integer(4), parameter :: max_size = 1024
    integer(4) :: map_array(0:max_size+10)
    character(10000) :: map_buf=""
    character(1024) :: host="unknown"
    character(1024) :: tag = ""
    character(11000) :: env_str = ""
    integer(4) :: num_threads = -1
    integer(4) :: y, x, tid, cpu, startx, endx
    character(100) :: omp_proc_bind_str="(none)"
    character(10000) :: omp_places_str="(none)"
    character(100) :: omp_num_threads_str="(none)"
    character(50) :: timestamp=""
    character(50) :: start_str=""
    character(50) :: end_str=""

    ! The hostnm call is not portable, so use HOST env variable if available
    ! Will be blank if not available, which is also OK
    call get_environment_variable("HOST", host)

    ! Get OMP env variables that could affect binding 
    call get_environment_variable("OMP_NUM_THREADS", omp_num_threads_str)
    call get_environment_variable("OMP_PROC_BIND", omp_proc_bind_str)
    call get_environment_variable("OMP_PLACES", omp_places_str)

    ! Add environment variables to env_str if set
    if (len(trim(omp_num_threads_str)) > 0) then
       env_str = trim(env_str) // " OMP_NUM_THREADS="//trim(omp_num_threads_str)
    end if

    if (len(trim(omp_proc_bind_str)) > 0) then
       env_str = trim(env_str) // " OMP_PROC_BIND="//trim(omp_proc_bind_str)
    end if

    if (len(trim(omp_places_str)) > 0) then
       env_str = trim(env_str) // " OMP_PLACES="//trim(omp_places_str)
    end if

    ! Generate a tag with hostname and MPI rank if provided
    if (rank >= 0) then
       write (tag, 30) "Rank ", rank, " ", trim(host)
30     format(a,i2,a,a)

    else
       tag=trim(host)
    end if

    ! Initialize cpu_id array to -1
    do y=0,max_size-1
       map_array(y) = -1
    end do
    ! Mark 1 past array as endpoint to simplify logic
    map_array(max_size) = -1

    !$omp parallel do private(y, tid, cpu)
    do y=0,1000000
       ! Mark what cpu_id used by iteration y
       tid = omp_get_thread_num()
       cpu = sched_getcpu()

       if ((cpu >= 0) .and. (cpu < max_size)) then
          ! Racy update but ok, just need it not to be -1
          if (map_array(cpu) < y) then
             map_array(cpu) = y
          end if
       else
          print '(a,i4)', "Unexpected tid ", tid , " cpu ", cpu
       end if
    end do
    !$omp end parallel do

    ! Create string with concise listing of cpus used
    num_threads=0
    map_buf="Map "
    x = 0
    do while (x <= max_size)
       startx=-1
       endx=-1
       ! Create string of cpu ids used by OpenMP threads
       if (map_array(x) .ne. -1) then
          ! Add comma if not first entry
          if (num_threads > 0) then
             map_buf=trim(map_buf)//","
          end if

          startx=x
          num_threads = num_threads + 1
          if (map_array(x+1) .ne. -1) then
             ! Count continuous thread numbers
             do while (map_array(x+1) .ne. -1)
                num_threads = num_threads + 1
                x = x + 1
             end do
             endx=x

             ! Convert ints to left justified strings suitable for printing
             write(start_str, *) startx
             write(end_str, *) endx
             start_str=adjustl(start_str)
             end_str=adjustl(end_str)

             ! Append range to map buf
             map_buf=trim(map_buf)//" "//trim(start_str)//"-"//trim(end_str)
          else
             ! Convert ints to left justified strings suitable for printing
             write(start_str, *) startx
             start_str=adjustl(start_str)

             ! Append single cpu id to map buf
             map_buf=trim(map_buf)//" "//trim(start_str)
          endif
       end if
       x = x + 1
    end do


    ! print out one line per process
    call get_timestamp_string(timestamp)
    print '(a, " ", a, " ", a, " Threads ",i3," ",a,a)', trim(timestamp), trim(variant_desc), trim(map_buf),&
         & num_threads, trim(tag), trim(env_str)
    flush(stdout)

    ! Return number of threads 
    print_openmp_mapping=num_threads
  end function print_openmp_mapping

  ! For non-MPI programs, turns into direct call to print_openmp_mapping
  !
  ! For MPI programs, syncs ranks, and print thread stats across ranks,
  ! in addition to calling print_openmp_mapping
  subroutine print_thread_stats (location)
    character(len=*), intent(in) :: location
    integer::intnumthreads
#ifdef USE_MPI
    real(8) numthreads, maxnumthreads, minnumthreads, sumnumthreads, avgnumthreads
    integer :: intmaxnumthreads, intminnumthreads

    ! Wait for all MPI ranks to complete
    call MPI_Barrier (MPI_COMM_WORLD, ierr)

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       print '(a, " Printing ", a," mapping (all tasks)")', trim(timestamp), trim(location)
       flush(stdout)
    end if

    ! Synchronize printf timestamps across MPI ranks
    last_timestamp = get_last_timestamp(secs_elapsed())
#endif

    ! Print OpenMP thread mapping, no MPI so pass in -1
    intnumthreads = print_openmp_mapping (location, -1)

#ifdef USE_MPI
    ! Make numthreads a double to make easier to work with
    numthreads = intnumthreads

    call MPI_Allreduce (numthreads, maxnumthreads, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce max numthreads returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    call MPI_Allreduce (numthreads, minnumthreads, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce min numthreads returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    call MPI_Allreduce (numthreads, sumnumthreads, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce sum numthreads returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if
    avgnumthreads = sumnumthreads / (1.0d+0 *numtasks)

    ! Don't print thread stats for no OpenMP case 
    if (rank == 0) then
       call get_timestamp_string(timestamp)
       intmaxnumthreads = maxnumthreads
       intminnumthreads = minnumthreads

       print '(a,a,a,a,i0,a,i0,a,f0.1,a,i0,a,f0.2,a,a)', trim(timestamp), &
            & " THREADSTATS ",trim(location), " ", numtasks, " tasks  min ", intminnumthreads,&
            & "  avg ", avgnumthreads, "  max ", intmaxnumthreads, &
            & "  maxdiff ", (maxnumthreads-minnumthreads)*100.0/minnumthreads, "% ", trim(variant_desc)
       flush(stdout)

    end if

    ! Sync all ranks on exit
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif

  end subroutine print_thread_stats
  ! end if if defined(TARGET_CPU_OMP) && !defined(NO_THREAD_MAP)
#endif

#if defined(USE_MPI) && !defined(NO_MPI_EXERCISERS)
  ! Do some short MPI stress tests that have exposed issues before (if in MPI mode) 

  ! This bcast test tends to break MPI when executables are not readable
  ! because this disables Linux's shared memory support (MPI has to detect
  ! and work around).
  ! Generally hits assertion or logs MPI error message.
  !
  ! (Typical test:  chmod -r ./a.out and run 2 or more tasks per node.)
  ! Recommend 8196, 131072.  Usually just 16432,16432 is sufficient to trigger.
  subroutine do_bcast_noshmmem_test (start_count_, end_count_, comm)
    use mpi
    integer, intent(in) :: start_count_, end_count_, comm
    integer :: start_count, end_count
    character, allocatable :: char_array(:)
    integer, allocatable :: int_array(:)
    real(8), allocatable :: double_array(:)
    character :: myval, rank0val
    integer :: intmyval, intrank0val
    real(8) :: doublemyval, doublerank0val
    integer :: err_count
    integer(8) :: i, j
    real(8) :: start_time, end_time, run_time

    ! Reset err_count each call
    err_count = 0

    ! Min acceptable start_count is 1 
    start_count = start_count_
    if (start_count < 1) then
       start_count = 1
    end if

    ! Min acceptable end_count is 'start_count' 
    end_count = end_count_
    if (end_count < start_count) then
       end_count = start_count
    end if

    ! Time all the tests for rough timing info 
    start_time = secs_elapsed()

    ! Allocate array of end_count+1 size to try different sizes of bcast
    allocate(char_array(0:end_count))
    allocate(int_array(0:end_count))
    allocate(double_array(0:end_count))

    ! Rank zero value will be 200, everyone else rank %128 
    rank0val = char(200)
    if (rank == 0) then
       myval = rank0val
    else
       myval = char(modulo(rank, 128))
    end if

    ! Convert myval and rank0val to int and double for those arrays
    intmyval=ichar(myval)
    doublemyval = intmyval
    intrank0val=ichar(rank0val)
    doublerank0val = intrank0val

    ! Data size dependent, 16384 reproduces for me
    i = start_count
    do while (i <= end_count)
       if (rank == 0) then
          call get_timestamp_string(timestamp)
          print '(a, " MPI_Bcast shmem exerciser with ", i6, " chars, ints, and doubles ", i0, " tasks")', trim(timestamp), &
               i, numtasks
          flush(stdout)
       endif

       ! Initialize data to different values on every rank
       char_array = myval
       int_array = intmyval
       double_array = doublemyval

       ! Test the different size/data type bcasts
       call MPI_Bcast(char_array, i, MPI_CHAR, 0, comm, ierr)

       call MPI_Bcast(int_array, i, MPI_INT, 0, comm, ierr)

       call MPI_Bcast(double_array, i, MPI_DOUBLE, 0, comm, ierr)

       ! Check that everyone got rank 0's value
       do j=0,i-1
          if (char_array(j) .ne. rank0val) then
             call get_timestamp_string(timestamp)
             print '(a," ",a,i0,a,i0, a,i0, a)', trim(timestamp), &
                  & "ERROR: MPI_Bcast exerciser: char_array[", j, "] = ",ichar(char_array(j)), " (", ichar(rank0val), " expected)"
             flush(stdout)
             err_count = err_count + 1
          end if
          if (int_array(j) .ne. intrank0val) then
             call get_timestamp_string(timestamp)
             print '(a," ",a,i0,a,i0, a,i0, a)', trim(timestamp), &
                  & "ERROR: MPI_Bcast exerciser: int_array[", j, "] = ",int_array(j), " (", intrank0val, " expected)"
             flush(stdout)
             err_count = err_count + 1
          end if
          if (double_array(j) .ne. doublerank0val) then
             call get_timestamp_string(timestamp)
             print '(a," ",a,i0,a,f0, a,f0, a)', trim(timestamp), &
                  & "ERROR: MPI_Bcast exerciser: double_array[", j, "] = ",double_array(j), " (", doublerank0val, " expected)"
             flush(stdout)
             err_count = err_count + 1
          end if
       end do

       if (err_count .ne. 0) then
          call get_timestamp_string(timestamp)
          print '(a," ", a, i0,a)', trim(timestamp), &
               & "ERROR: MPI_Bcast exercises ", err_count, " failures detected.  Exiting"
          flush(stdout)
          call die()
       end if


       ! Double each iteration
       i = i * 2
    end do


    ! Free buffer arrays
    deallocate(char_array)
    deallocate(int_array)
    deallocate(double_array)

    ! Time all the tests for rough timing info
    end_time = secs_elapsed()
    run_time = end_time - start_time

    if (rank == 0) then
       write (sec_str, 66) run_time
66     format(f18.4)
       sec_str=adjustl(sec_str)
       call get_timestamp_string(timestamp)
       print '(a, " MPI_Bcast exerciser total RUNTIME ", a, " s ", i0, " - ", i0, " ints ", i0, " tasks")', trim(timestamp), &
            trim(sec_str), start_count, end_count,  numtasks
       flush(stdout)
    end if


  end subroutine do_bcast_noshmmem_test

  ! Several MPIs have gotten intermittent wrong answers (racy) with the the MPI_Allreduce MAX
  ! of array values in this exerciser at scale.   Attempt to scale up to larger arrays at small task count
  subroutine do_allreduce_exerciser (tries, start_target_size_, end_target_size_, comm)
    use mpi
    integer, intent(in) ::  tries,  start_target_size_, end_target_size_, comm

    integer :: start_target_size, end_target_size
    integer, allocatable :: send_array(:), reduced_array(:)
    integer :: attempt, i
    integer :: max1, max1o, max2, max2o, max3, max3o
    integer :: err_count
    integer :: ints_per_task
    integer :: target_size, aim_for_size
    real(8) :: start_time, end_time, run_time
    integer :: expected_value

    ! Want at least one int per task and to run the test at least once
    start_target_size = start_target_size_
    if (start_target_size < numtasks) then
       start_target_size = numtasks
    endif

    ! Want the test to run at least once, so fix end_target_size if needed
    end_target_size = end_target_size_
    if (end_target_size < start_target_size) then
       end_target_size = start_target_size
    endif

    ! Time all the tests for rough timing info 
    start_time = secs_elapsed()

    ! Test target_sizes in range, doubling every time
    aim_for_size=start_target_size
    do while (aim_for_size <= end_target_size) 

       ! Every task gets same sized slice of array, may shrink target_size 
       ints_per_task = aim_for_size/numtasks

       ! Get as big as we can with same size per task
       target_size = numtasks * ints_per_task

       ! Allocate send and reduction array
       allocate(send_array(0:target_size-1))
       allocate(reduced_array(0:target_size-1))

       if (rank == 0) then
          call get_timestamp_string(timestamp)
          print '(a, " MPI_Allreduce exerciser total with ", i6, " ints ", i0, " iterations ", i0, " tasks")', trim(timestamp), &
               target_size, tries, numtasks
          flush(stdout)
       endif

       ! Initialize everything to negative task number so we can track down who's data we got.
       ! Use negative rank so works with more than 1000000 tasks
       send_array = -rank

       do attempt=0,tries-1
          ! Initial destination array to -1
          reduced_array = -1

          ! Set send_array at range assigned to each index to rank + 1000000.
          ! At the end of the MAX reduction, every index should be that index + 1000000.
          ! If not, it will tell us which data we actually got.
          do i=rank*ints_per_task,(((rank+1)*ints_per_task)-1)
             send_array(i) = rank + 1000000
          end do

          ! Create similar MPI noise as original code, do one small allreduce before
          max1 = rank
          call MPI_Allreduce (max1, max1o, 1, MPI_INT, MPI_MAX,  comm, ierr)
          if (ierr .ne. MPI_SUCCESS) then
             call get_timestamp_string(timestamp)
             print '(a," ", a, i0," ",a)', trim(timestamp), &
                  & "Error: MPI_Allreduce max1 returned ", ierr, trim(variant_desc)
             flush(stdout)
             call die()
          end if

          ! Do reduce with MAX, so should have every rank's initialization
          ! just in their range.
          call MPI_Allreduce (send_array, reduced_array, target_size, MPI_INT, MPI_MAX, comm, ierr)
          if (ierr .ne. MPI_SUCCESS) then
             call get_timestamp_string(timestamp)
             print '(a," ", a, i0," ",a)', trim(timestamp), &
                  & "Error: MPI_Allreduce send_array returned ", ierr, trim(variant_desc)
             flush(stdout)
             call die()
          end if

          max2 = 1 
          call MPI_Allreduce (max2, max2o, 1, MPI_INT, MPI_MAX,  comm, ierr)
          if (ierr .ne. MPI_SUCCESS) then
             call get_timestamp_string(timestamp)
             print '(a," ", a, i0," ",a)', trim(timestamp), &
                  & "Error: MPI_Allreduce max2 returned ", ierr, trim(variant_desc)
             flush(stdout)
             call die()
          end if

          max3 = rank/2
          call MPI_Allreduce (max3, max3o, 1, MPI_INT, MPI_MAX,  comm, ierr)
          if (ierr .ne. MPI_SUCCESS) then
             call get_timestamp_string(timestamp)
             print '(a," ", a, i0," ",a)', trim(timestamp), &
                  & "Error: MPI_Allreduce max3 returned ", ierr, trim(variant_desc)
             flush(stdout)
             call die()
          end if

          ! Expect index range to match value if reduction done properly
          err_count = 0
          do i=0, target_size-1
             ! Each rank gets a range of value here
             expected_value = (i/ints_per_task) + 1000000

             if (reduced_array(i) .ne. expected_value) then
                ! Only print at most 5 warnings per rank
                if (err_count < 5) then
                   call get_timestamp_string(timestamp)
                   print '(a," ",a,i0,a,i0, a,i0, a)', trim(timestamp), &
                        & "ERROR: MPI_Allreduce exerciser: reduced_array[", i, "] = ", &
                        & reduced_array(i), " (", expected_value, " expected)"
                   flush(stdout)
                end if
                if (err_count == 5) then
                   call get_timestamp_string(timestamp)
                   print '(a," ", a,a)', trim(timestamp), &
                        & "ERROR: MPI_Allreduce exerciser REMAINING ERROR MESSAGES SUPPRESSED! ", trim(variant_desc)
                   flush(stdout)
                end if
                err_count = err_count + 1
             end if

          end do
          if (err_count .ne. 0) then
             call get_timestamp_string(timestamp)
             print '(a," ", a, i0,a)', trim(timestamp), &
                  & "ERROR: MPI_Allreduce exercises ", err_count, " failures detected.  Exiting"
             flush(stdout)
             call die()
          end if
       end do

       ! Free memory 
       deallocate(send_array)
       deallocate(reduced_array)

       ! double size every time
       aim_for_size = aim_for_size * 2
    end do

    ! Time all the tests for rough timing info
    end_time = secs_elapsed()
    run_time = end_time - start_time

    if (rank == 0) then
       write (sec_str, 66) run_time
66     format(f18.4)
       sec_str=adjustl(sec_str)
       call get_timestamp_string(timestamp)
       print '(a, " MPI_Allreduce exerciser total RUNTIME ", a, " s ", i0, " - ", i0, & 
            &" ints ", i0, " iterations ", i0, " tasks")', trim(timestamp), &
            & trim(sec_str), start_target_size, end_target_size, tries, numtasks
       flush(stdout)
    end if
  end subroutine do_allreduce_exerciser

  ! Main stress test driver.  Individual stress tests above
  subroutine mpi_exerciser_driver (location)
    use mpi
    character(len=*), intent(in) :: location
    real(8) :: start_time, end_time, run_time
    integer :: subcomm

    ! Time all the tests for rough timing info
    start_time = secs_elapsed()

    ! Sync all the MPI ranks before starting
    call MPI_Barrier (MPI_COMM_WORLD, ierr)

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       print '(a, " Starting ", a," MPI exercisers  ", i0, " tasks ", a)', trim(timestamp), &
            & trim(location), numtasks, trim(variant_desc)
       flush(stdout)
    end if

    ! Synchronize printf timestamps across MPI ranks
    last_timestamp = get_last_timestamp(secs_elapsed())

    call MPI_Comm_dup(MPI_COMM_WORLD, subcomm, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: mpi_exerciser_driver: Failure in MPI_Comm_dup! ierr=", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    ! Do bcasts that usually use shared memory, useful for testing
    ! cases where executable is not readable (chmod -r ./a.out)
    call do_bcast_noshmmem_test (8192, 131072, subcomm)

    ! This allreduce test has broken several MPIs at scale.
    ! Scaled up to bigger arrays to try to trigger
    ! same issues even at small scales.
    call do_allreduce_exerciser (2, 8192, 131072, subcomm)

    ! Free sub communicator
    call MPI_Comm_free (subcomm, ierr)

    ! Time all the tests for rough timing info
    end_time = secs_elapsed()
    run_time = end_time - start_time

    if (rank == 0) then
       write (sec_str, 66) run_time
66     format(f18.4)
       sec_str=adjustl(sec_str)
       call get_timestamp_string(timestamp)
       print '(a, " Finished ", a," MPI exercisers RUNTIME ", a, " s ", i0, " tasks ", a)', trim(timestamp), &
            & trim(location), trim(sec_str), numtasks, trim(variant_desc)
       flush(stdout)
    end if

  end subroutine mpi_exerciser_driver

  ! end if defined(USE_MPI) && !defined(NO_MPI_EXERCISERS)
#endif

#ifdef TARGET_GPU_OMP
  ! If using OpenMP offloading, make sure GPU works before doing test 
  subroutine verify_gpu_openmp(gpu_id)
    use omp_lib
    integer, intent(in) :: gpu_id

    character(50) :: mpi_desc=""

    ! If using GPU, make sure GPU OpenMP gpu offloading works before doing test 
    integer:: runningOnGPU

#ifdef USE_MPI
    ! indicate MPI used 
    mpi_desc=" (all tasks)"

    ! Sync all the MPI ranks before selecting GPU 
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       print '(a," Selecting GPU ",i0, " as default device",a)', trim(timestamp), gpu_id, trim(mpi_desc)
       flush(stdout)
    end if

    ! Pick GPU to use to exercise selection call 
    call omp_set_default_device(gpu_id)

#ifdef USE_MPI
    ! Sync all the MPI ranks before printing start message 
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       print '(a," Launching OpenMP GPU test kernel",a)', trim(timestamp), trim(mpi_desc)
       flush(stdout)
    end if
    ! Test if GPU is available using OpenMP4.5 legal code 
    runningOnGPU = 0
    !$omp target map(from:runningOnGPU)
    if (.not. omp_is_initial_device()) then
       runningOnGPU = 1
    else
       runningOnGPU = 2
    end if
    !$omp end target

    ! If still running on CPU, GPU must not be available, punt 
    if (runningOnGPU .ne. 1) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "ERROR: OpenMP GPU test kernel did NOT run on GPU ", gpu_id, trim(variant_desc)
       flush(stdout)
       call die()
    end if

#ifdef USE_MPI
    ! Sync all the MPI ranks before starting 
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       print '(a," Verified OpenMP target test kernel ran on GPU",a)', trim(timestamp), trim(mpi_desc)
       flush(stdout)
    end if
  end subroutine verify_gpu_openmp
#endif


  !Print kernel runtime stats and aggregate across MPI processes if necessary.
  !Prints one liner if not using MPI
  subroutine print_runtime_stats(iterations, kernel_mem_used, kernel_runtime, transfer_runtime)
    ! Catch misspelled variables
    implicit none
    integer(8), intent(in) :: iterations
    real(8), intent(in) :: kernel_mem_used
    real(8), intent(in) :: kernel_runtime, transfer_runtime

#ifdef USE_MPI
    real(8):: kernel_maxruntime, kernel_minruntime, kernel_sumruntime, kernel_avgruntime, kernel_variation
    real(8):: transfer_maxruntime, transfer_minruntime, transfer_sumruntime, transfer_avgruntime, transfer_variation
    ! Get rank and numtasks from parent

    if (rank == 0) then
       if (numtasks > 1) then
          ! Print separator before and after output with function name
          call get_timestamp_string(timestamp)
          print '(a," Collecting and aggregating kernel runtimes across MPI ranks")', trim(timestamp)
          flush(stdout)
       end if
    end if

    call MPI_Allreduce (kernel_runtime, kernel_maxruntime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce max runtime returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    call MPI_Allreduce (kernel_runtime, kernel_minruntime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce min runtime returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    call MPI_Allreduce (kernel_runtime, kernel_sumruntime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce sum runtime returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if
    kernel_avgruntime = kernel_sumruntime / (1.0d+0*numtasks)
    if (kernel_minruntime > 0.0001) then
       kernel_variation = (kernel_maxruntime - kernel_minruntime)*100.0/kernel_minruntime
    else
       kernel_variation = (kernel_maxruntime - kernel_minruntime)*100.0/0.0001
    end if

    call MPI_Allreduce (transfer_runtime, transfer_maxruntime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce max runtime returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    call MPI_Allreduce (transfer_runtime, transfer_minruntime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce min runtime returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    call MPI_Allreduce (transfer_runtime, transfer_sumruntime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce sum runtime returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if
    transfer_avgruntime = transfer_sumruntime / (1.0d+0*numtasks)
    if (transfer_minruntime > 0.0001) then
       transfer_variation = (transfer_maxruntime - transfer_minruntime)*100.0/transfer_minruntime
    else
       transfer_variation = (transfer_maxruntime - transfer_minruntime)*100.0/0.0001
    end if

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       if (numtasks > 1) then
          print '(a,a,f7.2,a,f7.2,ai0,a,i0,a,a,a,a)', trim(timestamp), &
               &" VARIATION kernel ", kernel_variation, "%  datatrans ", transfer_variation, "% ", &
               & numtasks, " tasks  Rush Larsen ", iterations, " ", trim(kernel_mem_used_str), &
               & " ", trim(variant_desc)


          write (sec_str, 61) kernel_minruntime
61        format(f18.4)
          sec_str=adjustl(sec_str)
          write (us_str, 62) kernel_minruntime*1000000.0_8/(1.0d+0*iterations)
62        format(f18.2)
          us_str=adjustl(us_str)
          write (transfer_str, 63) transfer_minruntime
63        format(f18.4)
          transfer_str=adjustl(transfer_str)
          print '(a,a,i0,a,a,a,a,a,a,a,a,a,a,a)', trim(timestamp), &
               &" MINSTATS   Rush Larsen ",  iterations, " ", trim(kernel_mem_used_str), "  ", &
               & trim(sec_str), " s  ", trim(us_str), " us/iter  ", trim(transfer_str), " s datatrans ", trim(variant_desc)

          write (sec_str, 61) kernel_avgruntime
          sec_str=adjustl(sec_str)
          write (us_str, 62) kernel_avgruntime*1000000.0_8/(1.0d+0*iterations)
          us_str=adjustl(us_str)
          write (transfer_str, 63) transfer_avgruntime
          transfer_str=adjustl(transfer_str)
          print '(a,a,i0,a,a,a,a,a,a,a,a,a,a,a)', trim(timestamp), &
               &" AVGSTATS   Rush Larsen ",  iterations, " ", trim(kernel_mem_used_str), "  ", &
               & trim(sec_str), " s  ", trim(us_str), " us/iter  ", trim(transfer_str), " s datatrans ", trim(variant_desc)

          write (sec_str, 61) kernel_maxruntime
          sec_str=adjustl(sec_str)
          write (us_str, 62) kernel_maxruntime*1000000.0_8/(1.0d+0*iterations)
          us_str=adjustl(us_str)
          write (transfer_str, 63) transfer_maxruntime
          transfer_str=adjustl(transfer_str)
          print '(a,a,i0,a,a,a,a,a,a,a,a,a,a,a)', trim(timestamp), &
               &" MAXSTATS   Rush Larsen ",  iterations, " ", trim(kernel_mem_used_str), "  ", &
               & trim(sec_str), " s  ", trim(us_str), " us/iter  ", trim(transfer_str), " s datatrans ", trim(variant_desc)
       end if

       ! Our apps run in lockstep, so MAX time drives cycle time, so we use max for RUSHSTATS
       ! That said, we don't do sync across tasks every iterations, so worse in real apps
       write (sec_str, 61) kernel_maxruntime
       sec_str=adjustl(sec_str)
       write (us_str, 62) kernel_maxruntime*1000000.0_8/(1.0d+0*iterations)
       us_str=adjustl(us_str)
       write (transfer_str, 63) transfer_maxruntime
       transfer_str=adjustl(transfer_str)
       print '(a,a,i0,a,a,a,a,a,a,a,a,a,a,a)', trim(timestamp), &
            &" RUSHSTATS  Rush Larsen ",  iterations, " ", trim(kernel_mem_used_str), "  ", &
            & trim(sec_str), " s  ", trim(us_str), " us/iter  ", trim(transfer_str), " s datatrans ", trim(variant_desc)

       flush(stdout)
    end if

    ! NO MPI CASE - print one line
#else
    ! Print time stats
    ! Convert runtime into same format as C using string manipulation
    write (sec_str, 63) kernel_runtime
63  format(f18.4)
    sec_str=adjustl(sec_str)
    write (us_str, 64) kernel_runtime*1000000.0_8/(1.0d+0*iterations)
64  format(f18.2)
    us_str=adjustl(us_str)
    print '(a,a,i0,a,a,a,a,a,a,a,a,a)', trim(timestamp), &
         &" RUSHSTATS  Rush Larsen ",  iterations, " ", trim(kernel_mem_used_str), "  ", &
         & trim(sec_str), " s  ", trim(us_str), " us/iter  ", trim(variant_desc)
    flush(stdout)
#endif


  end subroutine print_runtime_stats

  ! Do sanity and consistency checks on all of m_gate. Including cross-rank if MPI mode
  ! Prints PASS or FAIL based on data check results
  ! If bad data found, will print up to 5 lines of debug info per MPI rank.
  ! Returns fail count so can be returned to caller.
  function data_check (m_gate, iterations, kernel_mem_used, nCells)
    ! Catch misspelled variables 
    implicit none
    real(8), dimension(0:), intent(inout) :: m_gate
    integer(8), intent(in) :: iterations
    real(8), intent(in) :: kernel_mem_used
    integer(8), intent(in) :: nCells
    integer(8) :: data_check ! Return value

    ! Local variables
    integer(8) :: fail_count

    ! In non-MPI mode, treat only process as rank 0
    integer :: rank = 0

    integer(8) :: i

#ifdef USE_MPI
    real(8) :: dfail_count, aggregate_fail_count=0 ! Make double so can use MPI to aggregate 
    real(8) :: checkval, rank0checkval, mincheckval, maxcheckval
    integer ret_code, ierr

    ! Get actual MPI rank if using MPI
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

    ! Sync all the MPI ranks before starting
    call MPI_Barrier (MPI_COMM_WORLD, ierr)

    ! Synchronize printf timestamps across MPI ranks
    last_timestamp = get_last_timestamp(secs_elapsed())
#endif

    ! Initialize variables on every entry
    fail_count = 0


    if (rank == 0) then
       ! Print separator before and after output with function name
       call get_timestamp_string(timestamp)
       print '(a," Starting data check for sanity and consistency")', trim(timestamp)
       flush(stdout)
    end if

    ! Sanity check that kernel not giving garbage
    ! Found m_gate[0] to be ~.0.506796353074569 after 1 iteration (really 2 with warmup)
    ! and converges to 0.996321172062538 after 100 iterations.  Make sure in that bounds
    ! for now.  With a little slop (~.000001) for now (not sure rounding error expected)
    if (m_gate(0) < 0.506796) then
       call get_timestamp_string(timestamp)
       print '(a," ",a,f17.15,a,a)', trim(timestamp), &
            & "ERROR Data sanity check m_gate[0]=", m_gate(0), " < 0.506796 (0.506796353074569 min expected value) ", &
            & trim(variant_desc)
       flush(stdout)
       fail_count = fail_count + 1

    else if (m_gate(0) > 0.996322) then
       call get_timestamp_string(timestamp)
       print '(a," ",a,f17.15,a,a)', trim(timestamp), &
            & "ERROR Data sanity check m_gate[0]=", m_gate(0), " > 0.996322 (0.996321172062538 max expected value) ", &
            & trim(variant_desc)
       flush(stdout)
       fail_count = fail_count + 1
    end if


    ! Every array entry should have the same value as m_gate[0], make sure that is true
    do i=1,nCells-1
       if (m_gate(i) .ne. m_gate(0)) then
          fail_count = fail_count + 1
          ! Only print at most 5 warnings per rank
          if (fail_count < 5) then
             call get_timestamp_string(timestamp)
             print '(a," ",a,i0,a,f17.15,a,f17.15,a)', trim(timestamp), &
                  & "ERROR Data sanity check m_gate[", i, "]=", m_gate(i), " != m_gate[0]=", m_gate(0), &
                  & trim(variant_desc)
             flush(stdout)
          end if
          if (fail_count == 5) then
             call get_timestamp_string(timestamp)
             print '(a," ", a,a)', trim(timestamp), &
                  & "ERROR Data consistency check REMAINING ERROR MESSAGES SUPPRESSED! ", trim(variant_desc)
             flush(stdout)
          end if
       end if
    end do

    ! Value looks ok, check all ranks match if using MPI
#ifdef USE_MPI

    ! With MPI, check that every rank gets same value for m_gate[0]
    ! Every task does its own checking of the rest of the values against m_gate[0]

    ! Get the kernel result we are checking
    checkval=m_gate(0)

    ! Everyone should check against rank 0's value
    if (rank == 0) then
       rank0checkval = checkval
    else
       rank0checkval = -1
    end if

    call MPI_Bcast(rank0checkval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Bcase rank 0's checkval returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    ! Everyone should check against rank 0's check value and print message on mismatch
    if (m_gate(0) .ne. rank0checkval) then
       call get_timestamp_string(timestamp)
       print '(a," ",a,i0,a,f17.15,a,f17.15,a)', trim(timestamp), &
            & "ERROR Data sanity check rank ", rank, "'s m_gate[0]=", m_gate(0), " != rank 0's m_gate[0]=", rank0checkval, &
            & trim(variant_desc)
       flush(stdout)
       fail_count = fail_count + 1
    end if

    ! Aggregate the fail count across all processes, convert to DOUBLE since MPI cannot sum MPI_LONG_INT
    dfail_count = fail_count
    aggregate_fail_count = -1
    call MPI_Allreduce (dfail_count, aggregate_fail_count, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce aggregate fail_count returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    ! Set all fail counts to aggregate fail count on all ranks (convert back to long int)
    fail_count = aggregate_fail_count

    ! Allow rank 0 to detect if there was data mismatch on a different rank
    call MPI_Allreduce (checkval, maxcheckval, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce max checkval returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    call MPI_Allreduce (checkval, mincheckval, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, ierr)
    if (ierr .ne. MPI_SUCCESS) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "Error: MPI_Allreduce min checkval returned ", ierr, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    ! If mismatches between ranks, min and max value values will be different
    if (maxcheckval .ne. mincheckval) then
       if (rank == 0) then
          call get_timestamp_string(timestamp)
          print '(a," ",a,f17.15,a,a)', trim(timestamp), &
               & "ERROR Data consistency check DETECTED MISMATCHES BETWEEN RANKS rank 0's m_gate[0]=", m_gate(0), " ", &
               & trim(variant_desc)
          flush(stdout)
       end if
    end if
#endif

    ! Print out summary PASSED or FAILED count from rank 0 only
    if (rank == 0) then
       if (fail_count == 0) then
          call get_timestamp_string(timestamp)
          print '(a,a,i0,a,a,a,f17.15,a,a)',&
               & trim(timestamp)," PASSED Data check ", iterations, " ", trim(kernel_mem_used_str), &
               & "  m_gate[0]=", m_gate(0), " ", trim(variant_desc)

          flush(stdout)
       else
          ! Convert kernel_mem_used to left justified string with leading 0
          call get_timestamp_string(timestamp)
          print '(a,a,i0,a,a,a,i0,a,f17.15,a,a)',&
               & trim(timestamp)," FAILED Data check ", iterations, " ", trim(kernel_mem_used_str), &
               & " with ", fail_count, " DATA CHECK ERRORS m_gate[0]=", m_gate(0), " ", trim(variant_desc)

          flush(stdout)
       end if
    end if

    data_check = fail_count 
  end function data_check
end function RUSH_LARSEN_VARIANT

#ifndef NO_MAIN
program rush_larsen_fort
  ! Only include mpi for variants that need it
#ifdef USE_MPI
  use mpi 
#endif
  ! Catch misspelled variables 
  implicit none
  interface 
     function RUSH_LARSEN_VARIANT(iterations, kernel_mem_used)
       integer(8),  intent(IN) :: iterations
       real(8), intent(IN) :: kernel_mem_used
       integer :: RUSH_LARSEN_VARIANT
     end function RUSH_LARSEN_VARIANT
  end interface

  ! For command line argument parsing
  character(1000) :: progName
  character(100) :: arg1char
  character(100) :: arg2char
  integer(8) :: max_iterations
  real(8) :: kernel_mem_used
  character(100) :: tag 
  integer :: rank = 0 !Rank will be 0 for the no MPI case 
  integer :: fail_count = 0

#ifdef USE_MPI
  integer :: ierr, rc

  call MPI_Init(ierr)
  if (ierr .ne. MPI_SUCCESS) then
     print *,'Error starting MPI program. Terminating.'
     call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
#endif


  call get_command_argument(0,progName)   !Get program name from arg 0

  !First, make sure the right number of inputs have been provided
  if(command_argument_count().ne.2) then
     if (rank == 0) then
        write(*,*) "Usage: ", trim(progName), "  Iterations  Kernel_GBs_used"
        write(*,*) " "
#ifdef TARGET_CPU_SERIAL
        write(*,*) "Measure serial baseline small:   ", trim(progName), " 100000 .00000001"
        write(*,*) "Measure serial baseline large:   ", trim(progName), "    100 10"
#else
        write(*,*) "Measure serial launch overhead:  env OMP_NUM_THREADS=1 ", trim(progName), " 100000 .00000001"
        write(*,*) "Measure launch overhead:         ", trim(progName), " 100000 .00000001"
        write(*,*) "Measure kernel performance:      ", trim(progName), "    100 10"
#endif
        write(*,*) " "
        write(*,*) trim(VERSION_STRING)
        write(*,*) " "
#ifdef COMPILERID
        write(*,*) "VARIANT_DESC: ", VARIANT_TAG, " [", COMPILERID, "]"
#else
        write(*,*) "VARIANT_DESC: ", VARIANT_TAG
#endif
#ifdef NO_MPI_EXERCISERS
        write(*,*) "Compiled with NO_MPI_EXERCISERS"
#endif
#ifdef NO_THREAD_MAP
        write(*,*) "Compiled with NO_THREAD_MAP"
#endif
        write(*,*) "  "
        write(*,*) "Questions? Contact John Gyllenhaal (gyllenhaal1@llnl.gov)\n"
     end if
     stop 1
  end if

  call get_command_argument(1,arg1char)   !read in the two values
  call get_command_argument(2,arg2char)

  read(arg1char,*)max_iterations                    !then, convert them to REALs
  read(arg2char,*)kernel_mem_used

  ! Don't print MPI_Init time for MPI version since the way I hid
  ! functions to enable interop makes timer routines hard to get to.

  ! Run the test
  fail_count =  RUSH_LARSEN_VARIANT (max_iterations, kernel_mem_used)

  ! Return 1 if data checks failed, before MPI Finalize
  if (fail_count .ne. 0) then
#ifdef USE_MPI
     call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
#else  
     stop 1
#endif
  end if

#ifdef USE_MPI
  call MPI_FINALIZE(ierr)
#endif

end program rush_larsen_fort
#endif
