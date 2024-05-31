# Makefile for compiling and running the program 'main.mlu' using cncc compiler with MTP_372 architecture

# Compiler and flags
CC = hipcc
CFLAGS = -std=c++17
OUTPUT_FLAG = -o

# Source file and output executable
SOURCE = main.cu.hip
EXECUTABLE = main.out

# Default target
all: $(EXECUTABLE)

# Compile the source code
$(EXECUTABLE): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) $(OUTPUT_FLAG) $@

# Ensure to use a literal tab character instead of spaces before each command line below.

# Clean up
clean:
	rm -f $(EXECUTABLE)

# Run the executable (optional)
run: $(EXECUTABLE)
	./$(EXECUTABLE)
