# demo for crnn

编译：
```
在根目录下执行make,生成静态库libcrnn.a, 可执行文件demo_crnn。
    
```
使用静态库（需要包含text_recognization.h）：
```
aarch64-himix100-linux-g++ demo_crnn.cpp libcrnn.a libncnn.a -O3 -mcpu=cortex-a53+simd -mcpu=cortex-a53+fp -o demo_crnn
```

执行效率(crnn单张图片识别)：
```
	双大核：210ms
	单大核：360ms
	双小核：670ms
	单小核：1250ms
```
