import sys
from tkinter import Y
from types import new_class

sys.dont_write_bytecode = True
import numpy as np
import math
import Metrics as metr  # 自己写的所有metric计算方式
import torch.nn.functional as F

import random
import os
import Models as models
import readData as rd
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
import SeqMIA as SeqMIA
import MetricSequence as MS

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from scipy.interpolate import interp1d

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import argparse


def load_data_for_trainortest(data_name):
    with np.load(data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in
                            range(len(f.files))]
    return train_x, train_y


def load_data_for_attack(data_name):
    with np.load(data_name) as f:
        train_x, train_y, data_label = [f['arr_%d' % i] for i in range(
            len(f.files))]
    return train_x, train_y, data_label


def getIndexByValue(dataYList, label):
    indexList = []
    for index, value in enumerate(dataYList):
        if value == label:
            indexList.append(index)
    return indexList


def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in
           dataToClip]
    return np.array(res)


def train_target_model(dataset, epochs=100, batch_size=100, learning_rate=0.001, l2_ratio=1e-7,
                       n_hidden=50, model='nn', datasetFlag='CIFAR10'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x, train_y, test_x, test_y = dataset

    # 获取分类的类别数量
    n_out = len(np.unique(train_y))
    if batch_size > len(train_y):
        batch_size = len(train_y)

    print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))

    if (datasetFlag == 'CIFAR10'):
        train_data = models.CIFARData(train_x, train_y)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loader_noShuffle = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_data = models.CIFARData(test_x, test_y)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        print("dataset error!")

    params = {}
    if (datasetFlag == 'CIFAR10'):
        params['task'] = 'cifar10'
        params['input_size'] = 32
        params['num_classes'] = 10
    else:
        print("datasetting error!")

    if model == 'vgg':
        print('Using vgg model...')
        # 继续组装params，字典类型
        params['conv_channels'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        params['fc_layers'] = [512, 512]
        params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        params['conv_batch_norm'] = True
        params['init_weights'] = True  # 运行初始化函数initialize_weights，在每个模型中都有。
        params['augment_training'] = True
        net = models.VGG(params)
        net = net.to(device)

    else:
        print("model type error!")

    net.train()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    if model == 'vgg':
        l2_ratio = 0.0005
        learning_rate = 0.1
        weight_decay_list = (param for name, param in net.named_parameters() if
                             name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                      {'params': no_decay_list, 'weight_decay': 0.}]
        momentum = 0.9
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=l2_ratio)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)  # 学习率衰减，每次衰减为原来的0.8

        if datasetFlag == 'CIFAR100':
            epochs = 100
    else:
        print("model type error!")
    print(
        'dataset: {},  model: {},  device: {},  epoch: {},  batch_size: {},   learning_rate: {},  l2_ratio: {}'.format(
            datasetFlag, model, device, epochs, batch_size, learning_rate, l2_ratio))
    count = 1
    print('Training...')
    for epoch in range(epochs):
        running_loss = 0
        for step, (X_vector, Y_vector) in enumerate(train_loader):
            X_vector = X_vector.to(device)
            Y_vector = Y_vector.to(device)
            output = net(X_vector)
            loss = criterion(output, Y_vector)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if optimizer.param_groups[0]['lr'] > 0.0005:
            scheduler.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, Loss: {:.5f},  lr: {}'.format(epoch + 1, running_loss, optimizer.param_groups[0]['lr']))

    print("Training finished!")
    pred_y = []
    net.eval()
    if batch_size > len(train_y):
        batch_size = len(train_y)
    for step, (X_vector, Y_vector) in enumerate(train_loader_noShuffle):
        # Y_vector = Y_vector.long()
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)
        output = net(X_vector)
        out_y = output.detach().cpu()  # 解绑梯度，才能被numpy之类操作。
        pred_y.append(np.argmax(out_y,
                                axis=1))  # 每次都添加一个Tensor到list中，所以pred_y是一个list。每个Tensor都有Batch_size个标签值，所以最后要用concatenate再去掉Tensor的壳。

    pred_y = np.concatenate(pred_y)
    print('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))

    print('Testing...')
    pred_y = []
    net.eval()
    if batch_size > len(test_y):
        batch_size = len(test_y)
    for step, (X_vector, Y_vector) in enumerate(test_loader):
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)

        output = net(X_vector)
        out_y = output.detach().cpu()
        pred_y.append(np.argmax(out_y,
                                axis=1))

    pred_y = np.concatenate(pred_y)
    print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('More detailed results:')
    print(classification_report(test_y, pred_y))

    attack_x, attack_y = [], []
    classification_y = []
    for step, (X_vector, Y_vector) in enumerate(train_loader_noShuffle):
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)
        output = net(X_vector)
        out_y = output.detach().cpu()
        softmax_y = softmax(out_y.numpy())
        Y_vector = Y_vector.detach().cpu()
        attack_x.append(softmax_y)
        attack_y.append(np.ones(len(Y_vector)))
        classification_y.append(Y_vector)

    for step, (X_vector, Y_vector) in enumerate(test_loader):
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)
        output = net(X_vector)
        out_y = output.detach().cpu()
        softmax_y = softmax(out_y.numpy())
        Y_vector = Y_vector.detach().cpu()
        attack_x.append(softmax_y)
        attack_y.append(np.zeros(len(Y_vector)))
        classification_y.append(Y_vector)
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classification_y = np.concatenate(classification_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classification_y = classification_y.astype('int32')

    return attack_x, attack_y, net, classification_y


def trainTarget(modelType, X, y,
                X_test=[], y_test=[],
                splitData=True,
                test_size=0.5,
                inepochs=50, batch_size=300,
                learning_rate=0.001, datasetFlag='CIFAR10'):
    if (splitData):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train = X
        y_train = y
    dataset = (X_train.astype(np.float32),
               y_train.astype(np.int32),
               X_test.astype(np.float32),
               y_test.astype(np.int32))
    attack_x, attack_y, theModel, classification_y = train_target_model(dataset=dataset, epochs=inepochs,
                                                                        batch_size=batch_size,
                                                                        learning_rate=learning_rate,
                                                                        n_hidden=128, l2_ratio=1e-07, model=modelType,
                                                                        datasetFlag=datasetFlag)

    return attack_x, attack_y, theModel, classification_y


def train_attack_model_RNN(dataset, epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                           n_hidden=50, model='rnn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_x, test_x = dataset
    num_classes = 2
    if batch_size > len(train_x):
        batch_size = len(train_x)
    print('Building model with {} training data, {} classes...'.format(len(train_x), num_classes))

    train_data = models.TrData(train_x)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=models.collate_fn)
    test_data = models.TrData(test_x)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=models.collate_fn)

    onetr = train_data[0]
    onepoint_size = onetr.size(1)
    input_size = onepoint_size - 1
    hidden_size = 50
    num_layers = 1

    if model == 'rnn':
        print('Using an RNN based model for attack...')
        net = models.lstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          num_classes=num_classes, batch_size=batch_size)
        net = net.to(device)
    elif model == 'rnnAttention':
        print('Using an RNN with atention model for attack...')
        net = models.LSTM_Attention(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    num_classes=num_classes, batch_size=batch_size)
        net = net.to(device)
    else:
        print('Using an error type for attack model...')

    net.train()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
    learning_rate = 0.01
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=l2_ratio)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    print(
        'model: {},  device: {},  epoch: {},  batch_size: {},   learning_rate: {},  l2_ratio: {}'.format(model, device,
                                                                                                         epochs,
                                                                                                         batch_size,
                                                                                                         learning_rate,
                                                                                                         l2_ratio))
    count = 1
    print('Training...')
    for epoch in range(epochs):
        running_loss = 0
        for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(train_loader):
            X_vector = X_vector.to(device)
            Y_vector = Y_vector.to(device)
            output, _ = net(X_vector, len_of_oneTr)
            output = output.squeeze(0)  # 第一维的1没有用，去掉。
            Y_vector = Y_vector.long()
            loss = criterion(output, Y_vector)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if optimizer.param_groups[0]['lr'] > 0.0005:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, Loss: {:.5f},  lr: {}'.format(epoch + 1, running_loss, optimizer.param_groups[0]['lr']))

    print("Training finished!")
    print('Testing...')
    pred_y = []
    pred_y_prob = []
    test_y = []
    hidden_outputs = []
    net.eval()
    if batch_size > len(test_x):
        batch_size = len(test_x)
    for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(test_loader):
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)
        output, hidden_output = net(X_vector, len_of_oneTr)
        output = output.squeeze(0)
        out_y = output.detach().cpu()
        pred_y.append(np.argmax(out_y,
                                axis=1))
        pred_y_prob.append(out_y[:, 1])
        test_y.append(Y_vector.detach().cpu())
        hidden_output = hidden_output.detach().cpu()
        hidden_output = np.squeeze(hidden_output)
        hidden_outputs.append(hidden_output)
    pred_y = np.concatenate(pred_y)
    pred_y_prob = np.concatenate(pred_y_prob)

    hidden_outputs = np.concatenate(hidden_outputs)

    test_y = np.concatenate(test_y)
    print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('More detailed results:')
    print(classification_report(test_y, pred_y))
    ROC_AUC_Result_logshow(test_y, pred_y_prob, reverse=False)

    return test_y, pred_y_prob, hidden_outputs


def AttackingWithShadowTraining_RNN(X_train, train_losses, X_test, test_losses, epochs=50, batch_size=20,
                                    modelType='rnn'):
    dataset = (X_train,
               X_test)
    l2_ratio = 0.0001
    targetY, pre_member_label, hidden_outputs = train_attack_model_RNN(dataset=dataset,
                                                                       epochs=epochs,
                                                                       batch_size=batch_size,
                                                                       learning_rate=0.01,
                                                                       n_hidden=64,
                                                                       l2_ratio=l2_ratio,
                                                                       model=modelType)

    return targetY, pre_member_label, hidden_outputs, test_losses


def ROC_AUC_Result_logshow(label_values, predict_values, reverse=False):
    if reverse:
        pos_label = 0
        print('AUC = {}'.format(1 - roc_auc_score(label_values, predict_values)))
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values, predict_values)))
    fpr, tpr, thresholds = roc_curve(label_values, predict_values,
                                     pos_label=pos_label)
    print("Thresholds are {}. The len of Thresholds is {}".format(thresholds, len(thresholds)))
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.loglog(fpr, tpr, 'b', label='AUC=%0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0.001, 1], [0.001, 1], 'r--')
    plt.xlim([0.001, 1.0])
    plt.ylim([0.001, 1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    ax = plt.gca()
    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    f = interp1d(xdata, ydata)
    fpr_0 = 0.001
    tpr_0 = f(fpr_0)
    print('TPR at 0.001 FPR is {}'.format(tpr_0))
    plt.show()


def softmax(x):
    shift = np.amax(x, axis=1)
    shift = shift.reshape(-1, 1)
    x = x - shift
    exp_values = np.exp(x)
    denominators = np.sum(np.exp(x), axis=1)
    softmax_values = (exp_values.T / denominators).T
    return softmax_values


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='attack dataset')
    parser.add_argument('--pathLoadData', type=str, default='./data/cifar-10-batches-py-official',
                        help='the load path of dataset')
    parser.add_argument('--classifierType', type=str, default='vgg', help='model type')
    parser.add_argument('--num_epoch', type=int, default=100, help='train epoch')
    parser.add_argument('--num_epoch_for_distillation', type=int, default=50, help='distillation epoch')
    parser.add_argument('--attack_epoch', type=int, default=150, help='attack epoch(RNN epoch)')
    parser.add_argument('--metricFlag', type=str, default='loss&max&sd&entropy&mentropy')
    parser.add_argument('--resultDataPath', type=str, default='./results/', help='the path of results')
    parser.add_argument('--preprocessData', type=bool, default=True, help='True:preprocess dataset')
    parser.add_argument('--trainTargetModel', type=bool, default=True, help='True:train target model')
    parser.add_argument('--trainShadowModel', type=bool, default=True, help='True:train shadow model')
    parser.add_argument('--distillTargetModel', type=bool, default=True, help='True:distill target model')
    parser.add_argument('--distillShadowModel', type=bool, default=True, help='True:distill shadow model')
    return parser.parse_args()


if __name__ == '__main__':
    args=set_args()
    dataset = args.dataset
    pathToLoadData = args.pathToLoadData
    classifierType = args.classifierType
    dataFolderPath = args.dataFolderPath

    num_epoch = args.num_epoch
    attack_epoch = args.attack_epoch
    num_epoch_for_distillation = args.num_epoch_for_distillation
    metricFlag = args.metricFlag
    resultDataPath = args.resultDataPath
    preprocessData = args.preprocessData
    trainTargetModel = args.trainTargetModel
    trainShadowModel = args.trainShadowModel
    distillTargetModel = args.distillTargetModel
    distillShadowModel = args.distillShadowModel

    try:
        os.makedirs(resultDataPath)
    except OSError:
        pass

    targetX, targetY, shadowX, shadowY, target_classification_y, shadow_classification_y, target_losses, shadow_losses = SeqMIA.generateAttackDataForSeqMIA(
        dataset, classifierType, dataFolderPath, pathToLoadData, num_epoch, preprocessData, trainTargetModel,
        trainShadowModel, topX=3, num_epoch_for_distillation=num_epoch_for_distillation,
        distillTargetModel=distillTargetModel, distillShadowModel=distillShadowModel, metricFlag=metricFlag)

    print("Attacking using SeqMIA...")
    print("metric:  {}".format(metricFlag))

    num_metrics = metricFlag.count('&') + 1

    if num_metrics == 1:
        targetData = MS.createLossTrajectories_Seq(targetX, targetY,
                                                   num_metrics)
        shadowData = MS.createLossTrajectories_Seq(shadowX, shadowY,
                                                   num_metrics)
    else:
        targetData = MS.createMetricSequences(targetX, targetY,
                                              num_metrics)
        shadowData = MS.createMetricSequences(shadowX, shadowY,
                                              num_metrics)
    modelType = 'rnnAttention'
    targetY, pre_member_label, hidden_outputs, losses = AttackingWithShadowTraining_RNN(shadowData, shadow_losses,
                                                                                        targetData, target_losses,
                                                                                        epochs=attack_epoch,
                                                                                        batch_size=100,
                                                                                        modelType=modelType)
    np.savez(resultDataPath + 'SeqMIA_{}_{}.npz'.format(modelType, metricFlag), targetY,
             pre_member_label)
