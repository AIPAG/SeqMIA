# Project introduction

  This is the official repository for the ACM CCS 2024 paper "SeqMIA: Sequential-Metric Based Membership Inference Attack" 
  by Hao Li, Zheng Li, Siyuan Wu, Chengrui Hu, Yutong Ye, Min Zhang, Dengguo Feng, and Yang Zhang.
  In this project, we demonstrate the Sequential-metric based Membership Inference Attack(SeqMIA) that recognizes and utilizes 
  integrated membership signal: the Pattern of Metric Sequence, derived from the various stages of model training.

# Environment dependencies

  pytorch:2.0.1+cu118
  
  python:3.8.17

  Note that we tested this under Windows. If you are on Linux, you may need to change the matplotlib plotting part.

# Directory structure

```
  ├── README.md
  
  ├── data
  
  │   ├── cifar-10-batches-py-official
  
  │   ├── CIFAR10
  
  │       ├── PreprocessedIncludingDistillationData
  
  ├── model_IncludingDistillation
  
  │   ├── CIFAR10
  
  │       ├── Shadow
  
  │       ├── Target
  
  ├── results
  
  ├── attackMethodsFramework.py
  
  ├── Metrics.py
  
  ├── MetricSequence.py
  
  ├── Models.py
  
  ├── readData.py
  
  ├── SeqMIA.py
  
  └── temp
```
  
# Supported Dataset and Model

  CIFAR10
  
  vgg-16
  
  Following this example, more datasets and models can be added to SeqMIA.
  
# Usage instructions

  First, please put the CIFAR10 dataset (downloaded from its official web, i.e. data_batch_1 to data_batch_5 and test_batch) into `cifar-10-batches-py-official`. Then, you can run `attackMethodsFramework.py` to start the entire attack process.

  You can set some hyperparameters in set_args() or change them on the command line as follows.
  
  1.The first time you run this code, please set `preprocessData` to `True`. And processed data will be in the `PreprocessedIncludingDistillationData` folder.
  
  2.Set `trainTargetModel` and `trainShadowModel` to `True` to train the target model and the shadow model when needed.
  
  3.Set `distillTargetModel` and `distillShadowModel` to `True` to distill the models when needed.

  For example, after placing the CIFAR10 dataset into `cifar-10-batches-py-official`, run the command `python attackMethodsFramework.py -preprocessData -trainTargetModel -trainShadowModel -distillTargetModel -distillShadowModel`.
  
# Authors

  This project is created by Lihao(ISCAS)

# Cite

  if you use SeqMIA for your research,please cite Our paper [XXX](https://www.XXX.com)
