# Makefile for compiling and running the program 'main.mlu' using cncc compiler with MTP_372 architecture

# Compiler and flags
CC = cncc
MLU_ARCH_FLAG = --bang-mlu-arch=mtp_372
OUTPUT_FLAG = -o

# Source file and output executable
SOURCE = main.mlu
EXECUTABLE = main.out

# Default target
all: $(EXECUTABLE)

# Compile the source code
$(EXECUTABLE): $(SOURCE)
	$(CC) $(MLU_ARCH_FLAG) $(SOURCE) $(OUTPUT_FLAG) $@

# Ensure to use a literal tab character instead of spaces before each command line below.

# Clean up
clean:
	rm -f $(EXECUTABLE)

# Run the executable (optional)
run: $(EXECUTABLE)
	./$(EXECUTABLE)
