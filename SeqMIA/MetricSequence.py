import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import Dataset  # 导入抽象类Dataset
import os
import pandas as pd
from sklearn import preprocessing
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import csv

import math
import random

import Models as models  # 自己写的所有的模型架构

import Metrics as metr  # 自己写的所有metric计算方式
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import attackMethodsFramework as att_frame


def createMetricSequences(targetX, targetY, num_metrics):
    privacyMetircs = np.split(targetX, num_metrics, axis=1)
    for i in range(0, num_metrics):
        privacyMetircs[i] = np.roll(privacyMetircs[i], -1, axis=1)
    trajectory_data = []
    for i in range(0, len(targetX)):
        oneTr = np.vstack((privacyMetircs[0][i], privacyMetircs[1][i], privacyMetircs[2][i], privacyMetircs[3][i],
                           privacyMetircs[4][i]))  # (5*51)
        oneTr = np.swapaxes(oneTr, 0, 1)
        oneTr = np.insert(oneTr, 0, targetY[i], axis=1)
        trajectory_data.append(torch.Tensor(oneTr))

    return trajectory_data


def createLossTrajectories_Seq(targetX, targetY, num_metrics):
    privacyMetircs = targetX
    privacyMetircs = np.roll(privacyMetircs, -1, axis=1)
    trajectory_data = []
    for i in range(0, len(targetX)):
        oneTr = privacyMetircs[i]
        oneTr = oneTr[:, np.newaxis]
        oneTr = np.insert(oneTr, 0, targetY[i], axis=1)
        trajectory_data.append(torch.Tensor(oneTr))

    return trajectory_data
