# Makefile for compiling and running the program 'main.mlu' using cncc compiler with MTP_372 architecture

# Compiler and flags
CC = clang++

OUTPUT_FLAG = -o

# Source file and output executable
SOURCE = main.cpp
EXECUTABLE = main.out

# Default target
all: $(EXECUTABLE)

CUDA      = yes
CUDA_ARCH = sm_80
HIP       = no
HIP_ARCH  = gfx906

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall
CFLAGS += -fsycl
ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH) \
            #--gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
            -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif


# Compile the source code
$(EXECUTABLE): $(SOURCE)
	$(CC) $(CFLAGS)  $(SOURCE) $(OUTPUT_FLAG) $@

# Ensure to use a literal tab character instead of spaces before each command line below.

# Clean up
clean:
	rm -f $(EXECUTABLE)

# Run the executable (optional)
run: $(EXECUTABLE)
	./$(EXECUTABLE)
