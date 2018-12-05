CFLAGS += -I$(PWD)/ncnn -I$(PWD)/dlib -L$(PWD) -lncnn -ldlib
CFLAGS += -O3 -fopenmp -mcpu=cortex-a53+simd -mcpu=cortex-a53+fp

all:interface_crnn edit_dis demo_crnn

interface_crnn:
	aarch64-himix100-linux-g++ interface_crnn.cpp $(CFLAGS) -c -o interface_crnn.cpp.o 
	ar r libcrnn.a interface_crnn.cpp.o

edit_dis:
	aarch64-himix100-linux-g++ edit_dis.cpp $(CFLAGS) -c -o edit_dis.cpp.o 
	ar r libcrnn.a edit_dis.cpp.o



demo_crnn:interface_crnn
	aarch64-himix100-linux-g++ demo_crnn.cpp libcrnn.a $(CFLAGS) -o demo_crnn 

