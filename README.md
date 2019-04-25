# demo for crnn（输入文件是图片二进制RGB数据）

测试：
```
测试代码在test目录下，使用的图片和相应的字典来自http://www.robots.ox.ac.uk/~vgg/data/text。
```
模型转换：
```
mxnet的权重转换（.bin）转换见https://github.com/kouxichao/ncnn，相应的mxnet2ncnn。
计算图.param转换目前需要手动调整（应该蛮简单的），并没有更改。

pytorch转换参考https://github.com/starimeL/PytorchConverter,此项目由pytorch模型转过来的(https://github.com/meijieru/crnn.pytorch)。
```

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
