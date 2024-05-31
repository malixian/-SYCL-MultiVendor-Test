# Define compiler and flags
CC := clang++
CFLAGS := -fsycl -fsycl-targets=mlisa-cambricon-bang

# Name of the executable and source file(s)
EXECUTABLE := add.out
SOURCES := add.cpp

# Default target
all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@

# Clean up target
clean:
	rm -f $(EXECUTABLE)

# Run the executable (optional)
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Phony target for better re-running behavior
.PHONY: clean
