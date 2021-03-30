# Example makefile, will have to adjust to your OpenMP GPU offload 
# compiler and the GPU targetted
# This targets AMD MI60 GPU with AMD's ROCM OpenMP compiler

# Example of creating compiler tag from compiler version info
CC := /opt/rocm-4.0.1/llvm/bin/clang++
TAG := $(shell which $(CC))

all:  rush_larsen_omp

rush_larsen_omp: rush_larsen_omp.cc
        # Safe to remove COMPILERID def or hardcode TAG
	$(CC) -o rush_larsen_omp -O3 "-DCOMPILERID=$(TAG)" -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 rush_larsen_omp.cc

clean:
	rm -f rush_larsen_omp rush_larsen_omp.o
