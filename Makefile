CFLAGS += -I$(PWD) -L$(PWD) -lvision -ldlib -lncnn
CFLAGS += -O3 -fopenmp -mcpu=cortex-a53+simd -mcpu=cortex-a53+fp
SRCS := $(wildcard *.cpp)

all:demo_crnn

demo_crnn:
	aarch64-himix100-linux-g++ $(SRCS) $(CFLAGS) -o demo_crnn        
