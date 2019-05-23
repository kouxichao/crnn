# demo for crnn

测试：
```
测试代码在test目录下，使用的图片和相应的字典来自http://www.robots.ox.ac.uk/~vgg/data/text。
```
模型转换：
```
mxnet的权重转换（.bin）转换见https://github.com/kouxichao/ncnn，相应的mxnet2ncnn。
计算图.param转换目前需要手动调整（应该蛮简单的），并没有更改。

pytorch转换参考https://github.com/starimeL/PytorchConverter,此项目由pytorch模型转过来的(https://github.com/meijieru/crnn.pytorch)。

我的转换代码：https://pan.baidu.com/s/1sx2U5SRz0gAl3GRW3i-IEQ;提取码：1ugk
param需要自己对照更改。使用pytorch-0.2可以完成转换，已测试。
```

# Compile
ubuntu or other platform compile:
```
依赖库：
	dlib库：需要自己编译生成dlib库文件，这里只用到dlib进行图片加载，当然也可以使用其他图片处理库如opencv，并更改相应函数。
	ncnn库：参照https://github.com/kouxichao/ncnn进行编译。
```

3559A compile：
```
在根目录下执行make,生成静态库libcrnn.a, 可执行文件demo_crnn。
    
使用静态库（需要包含text_recognization.h）：

aarch64-himix100-linux-g++ demo_crnn.cpp libcrnn.a libncnn.a -O3 -mcpu=cortex-a53+simd -mcpu=cortex-a53+fp -o demo_crnn


执行效率(crnn单张图片识别)：

	双大核：210ms
	单大核：360ms
	双小核：670ms
	单小核：1250ms
```
