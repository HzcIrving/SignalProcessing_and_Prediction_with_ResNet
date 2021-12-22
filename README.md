### Signal Processing and Prediction with 1D-ResNet18/34 

## CHVer Description 中文版描述 
一个小项目，主要功能如下：
1. 通过.lmdb创建自己的一维数据集 
  -  使用的是基于Protobuf的lmdb包。
  -  Pytorch可以方便的进行Dataloader的设置。 
2. 模型，包括了1DResNet18、1DResNet34、MultiScale 1D ResNet
  -  都是处理一维信号数据的工具Backbone。 
感兴趣可以自己Clone下来玩一玩。 

## EngVer Description 
This is a small project for 1D signal data processing. The main functions are as follows: 
1. 1st, you can create your own ".lmdb" standard dataset. ( only 1D signal ) 
  - We utilize the lmdb generating package ( including in our project ) on the basis of "Protobuf". 
  - Pytorch contains standard API for lmdb dataset. 
2.  Backbone model including 1DResNet18/34 and Multiscale 1D ResNet. 
If you are interested in this project, you can have a try. 
