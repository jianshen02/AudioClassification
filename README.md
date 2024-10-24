# AudioClassification

## 简介

构建了模型用于对UrbanSound8K数据集进行分类，并考虑使用多线程进行推理加快推理过程。

## 文件说明

- CNN_train.py: 训练简易CNN模型的源码
- CNN_infer.py: 使用简易CNN模型进行推理的源码
- YAMNet_train.py: 微调YAMNet模型的源码
- YAMNet_infer.py: 使用微调好的YAMNet模型进行推理的源码
- YAMNet_parallel_infer.py: 利用多线程对使用微调好的YAMNet模型的推理进行加速的源码
- testGPU.py: 检测设备是否使用GPU

## 使用方法

- 上述文件共包含两个模型，先train得到模型文件后，使用infer进行推理即可。
