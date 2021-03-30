
# Example of creating compiler tag from compiler version info
TAG := $(shell which hipcc)

all:  rush_larsen_hip

rush_larsen_hip: rush_larsen_hip.cc
        # Safe to remove COMPILERID def or hardcode TAG
	hipcc -o rush_larsen_hip -O3 "-DCOMPILERID=$(TAG)" rush_larsen_hip.cc

clean:
	rm -f rush_larsen_hip rush_larsen_hip.o
