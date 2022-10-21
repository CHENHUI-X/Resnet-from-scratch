# Resnet-from-scratch
Implement a simple Resnet according to the tutorial
- 根据教程使用pytorch实现Resnet
  - 链接 ： https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
- 微调了教程中的一些代码，比如validationset 实际和 trainingset来自一个batch
- 使用pytorch自带的dataloader下载CIFAR-10数据
- 添加了部分注释，标识了layer的输出shape
- 添加了visualization part
---
2022年10月21日更新
`Resnet_labml.py` 文件结合了Labmlai的writer库进行实现.
