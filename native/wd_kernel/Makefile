# Detect OS and set platform-specific output name and flags
UNAME := $(shell uname -s 2>/dev/null || echo Windows)
OUT_DIR ?= ../../build/native/wd_kernel

ifeq ($(UNAME), Darwin)
    # macOS: clang requires libomp installed via Homebrew
    # brew install libomp
    CC      = gcc
    TARGET  = wd_kernel.so
    CFLAGS  = -shared -fPIC -O3 -Xpreprocessor -fopenmp -lomp -lm
    LDFLAGS = -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include
else ifeq ($(UNAME), Windows)
    # Native Windows with MinGW-w64 (gcc from https://winlibs.com or MSYS2)
    CC      = gcc
    TARGET  = wd_kernel.dll
    CFLAGS  = -shared -O3 -fopenmp -lm
    LDFLAGS =
else
    # Linux (gcc with OpenMP)
    CC      = gcc
    TARGET  = wd_kernel.so
    CFLAGS  = -shared -fPIC -O3 -fopenmp -lm
    LDFLAGS =
endif

TARGET_PATH := $(OUT_DIR)/$(TARGET)

all: $(TARGET_PATH)

$(TARGET_PATH): wd_kernel.c uthash.h
	mkdir -p $(OUT_DIR)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $(TARGET_PATH) wd_kernel.c

clean:
	rm -f $(OUT_DIR)/wd_kernel.so $(OUT_DIR)/wd_kernel.dll $(OUT_DIR)/wd_kernel.dylib
	rm -f wd_kernel.so wd_kernel.dll wd_kernel.dylib
