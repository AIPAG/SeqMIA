Project introduction
This is the official repository for the ACM CCS 2024 paper "SeqMIA: Sequential-Metric Based Membership Inference Attack" 
by Hao Li, Zheng Li, Siyuan Wu, Chengrui Hu, Yutong Ye, Min Zhang, Dengguo Feng, and Yang Zhang.
In this project, we demonstrate the Sequential-metric based Membership Inference Attack(SeqMIA) that recognizes and utilizes 
integrated membership signal: the Pattern of Metric Sequence, derived from the various stages of model training.
Environment dependencies
pytorch:2.0.1+cu118
python:3.8.17
Directory structure
├── ReadMe.md

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
Supported Dataset and Model
CIFAR10
vgg-16
Following this example, more datasets and models can be added to SeqMIA.
Usage instructions
run attackMethodsFramework.py to start the entire attack process.

1.set your argparse in set_args() or change in the command line with additional parameters
2.To run this code, you first need to have the processed data. Please set `preprocessData` to `True`.
3.Set `trainTargetModel` and `trainShadowModel` to True to train the target model and the shadow model.
4.Set `distillTargetModel` and `distillShadowModel` to `True` to distill the models.
Authors
This project is created by Lihao(ISCAS)

Cite
if you use SeqMIA for your research,please cite xxx.
