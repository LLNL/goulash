# Example makefile, will have to adjust to your OpenMP GPU offload 
# compiler and the GPU targetted
# This targets AMD MI60 GPU with AMD's ROCM OpenMP compiler

# Example of creating compiler tag from compiler version info
CC := /opt/rocm-4.0.1/llvm/bin/clang++
TAG := $(shell which $(CC))


rush_larsen_cpu: rush_larsen_cpu.cc
        # Safe to remove COMPILERID def or hardcode TAG
	$(CC) -o rush_larsen_cpu -O3 "-DCOMPILERID=$(TAG)" -fopenmp rush_larsen_cpu.cc

all:  rush_larsen_cpu rush_larsen_serial

rush_larsen_serial: rush_larsen_cpu.cc
	$(CC) -o rush_larsen_serial -O3 "-DCOMPILERID=$(TAG)" -DNOOMP rush_larsen_cpu.cc

clean:
	rm -f rush_larsen_cpu rush_larsen_serial rush_larsen_cpu.o
