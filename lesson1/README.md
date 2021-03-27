# README



人工智能初体验 MNIST 手写训练集识别

基础环境配置 anaconda + pycharm 

https://blog.csdn.net/sunshine_lyn/article/details/81158855

一、配置环境

为了之后方便使用不同的框架，我们采用anaconda虚拟环境1、

1、创建一个虚拟环境

```
conda create -n pytorch python=3.8
```

按照提示操作，创建完成即可

2、换国内镜像源

https://blog.csdn.net/yst990102/article/details/106730382

按照操作替换即可 （有时清华源用的人多所以也会出现超时情况 我自己用的豆瓣，很快）

3、安装pytorch

(1)查看电脑的显卡版本 

​	桌面右击显卡驱动 -> 系统信息 -> 第三行有CUDA版本号

(2)在 pytorch.org 中按照对应的配置选择下载命令，在anaconda命令行中运行该命令，下载完毕即可

二、新建工程

1、在pycharm新建工程，编译器选择刚刚安装的虚拟环境（在conda environment的existing environment中）

2、导入mnist.py文件 运行

可能出现的错误

（1）mnist 数据集无法下载 HTTP503 ERROR

​			原因：mnist数据集的下载网站不稳定

​			解决方案：先在群里下载mnist数据集，然后在Dataloader中的路径改为mnist的路径即可

（2）没有matplotlib库

​			解决方案：在anaconda命令行中

```
pip install matplotlib
```

三、使用模型

1、在工程中导入model_use.py

2、下载opencv库 具体命令为

```
pip install opencv-python
```

3、运行程序 结果是image文件夹中识别的概率向量和结果

可能出现的错误

（1）没有mnist模块

原因：在model_use.py中使用的mnist.py中的MLP类 由于个人命名不同导致错误

解决方案：按照MLP类所在的文件导入 具体格式为

```python
from 文件名.py import MLP
```



如有问题欢迎在群里交流或者私戳我的qq！

千里之行，始于足下。我们已经有了一个好的开端！大家加油！