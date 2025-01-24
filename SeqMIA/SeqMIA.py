import torch
import attackMethodsFramework as att_frame
import numpy as np
import Models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import Metrics as metr
import readData as rd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import os
import math
import random

def shuffleAndSplitDataByClass_distillation(dataX, dataY,cluster):
    
    n_class = np.unique(dataY)
    cluster_for_one_class = math.floor(cluster/len(n_class))
    toTrainDataIndexTotal = []
    shadowDataIndexTotal = []
    toTestDataIndexTotal = []
    shadowTestDataIndexTotal = []
    distillationDataIndexTotal = []
    dataYList = dataY.tolist()
    print("total number of samples: {}, total number of classes:{}, number of samples per class:{}".format(len(dataYList), len(n_class), cluster_for_one_class))
    print("number of distillation samples is {}".format(len(dataYList)-4*cluster_for_one_class*len(n_class)))
    for label in n_class:		
        cluster_for_one_class = math.floor(cluster/len(n_class))
        dataIndex = att_frame.getIndexByValue(dataYList, label)

        random.shuffle(dataIndex)
        
        if math.floor(len(dataIndex)/5) < cluster_for_one_class:
            cluster_for_one_class = math.floor(len(dataIndex)/5)
        toTrainDataIndex  = np.array(dataIndex[:cluster_for_one_class])
        shadowDataIndex  = np.array(dataIndex[cluster_for_one_class:cluster_for_one_class*2])
        toTestDataIndex  = np.array(dataIndex[cluster_for_one_class*2:cluster_for_one_class*3])
        shadowTestDataIndex  = np.array(dataIndex[cluster_for_one_class*3:cluster_for_one_class*4])
        distillationDataIndex = np.array(dataIndex[cluster_for_one_class*4:])

        print("class {} has {}, {}, {}, {} samples for targetModel and shadowModel, {} samples for distillation.".format(label, len(toTrainDataIndex), len(shadowDataIndex), len(toTestDataIndex), len(shadowTestDataIndex), len(distillationDataIndex)))

        toTrainDataIndexTotal.append(toTrainDataIndex)
        shadowDataIndexTotal.append(shadowDataIndex)
        toTestDataIndexTotal.append(toTestDataIndex)
        shadowTestDataIndexTotal.append(shadowTestDataIndex)
        distillationDataIndexTotal.append(distillationDataIndex)
    toTrainDataIndexTotal = np.concatenate(toTrainDataIndexTotal).astype(np.int64)
    shadowDataIndexTotal = np.concatenate(shadowDataIndexTotal).astype(np.int64)
    toTestDataIndexTotal = np.concatenate(toTestDataIndexTotal).astype(np.int64)
    shadowTestDataIndexTotal = np.concatenate(shadowTestDataIndexTotal).astype(np.int64)
    distillationDataIndexTotal = np.concatenate(distillationDataIndexTotal).astype(np.int64)
    random.shuffle(toTrainDataIndexTotal)
    random.shuffle(shadowDataIndexTotal)
    random.shuffle(toTestDataIndexTotal)
    random.shuffle(shadowTestDataIndexTotal)
    random.shuffle(distillationDataIndexTotal)


    toTrainData = np.array(dataX[toTrainDataIndexTotal])
    toTrainLabel = np.array(dataY[toTrainDataIndexTotal])
    
    shadowData  = np.array(dataX[shadowDataIndexTotal])
    shadowLabel = np.array(dataY[shadowDataIndexTotal])
    
    toTestData  = np.array(dataX[toTestDataIndexTotal])
    toTestLabel = np.array(dataY[toTestDataIndexTotal])
    
    shadowTestData  = np.array(dataX[shadowTestDataIndexTotal])
    shadowTestLabel = np.array(dataY[shadowTestDataIndexTotal])

    distillationData  = np.array(dataX[distillationDataIndexTotal])
    distillationDataLabel = np.array(dataY[distillationDataIndexTotal])

    return toTrainData, toTrainLabel,   shadowData,shadowLabel,    toTestData,toTestLabel,      shadowTestData,shadowTestLabel,    distillationData, distillationDataLabel


def get_file_name(path):
    model_file_name = []
    final_file_name = []
    files = os.listdir(path)  # 采用listdir来读取所有文件
    for i in files:
        model_file_name.append(i.replace("distilledModel_",""))
  
    model_file_name.sort(key=lambda x: int(x[:x.find(".")]))
    for j in model_file_name:
        final_file_name.append("distilledModel_"+j)
    return final_file_name

def initializeTargetModelforDistill(dataset,num_epoch,dataFolderPath= './data/',modelFolderPath = './model_IncludingDistillation/',classifierType = 'cnn', num_epoch_for_distillation = 50):

    dataPath = dataFolderPath+dataset+'/PreprocessedIncludingDistillationData'
    attackerModelDataPath = dataFolderPath+dataset+'/attackerModelDataIncludingDistillation'
    modelPath = modelFolderPath + dataset
    try:
        os.makedirs(attackerModelDataPath)
    except OSError as ee:
        pass
    try:
        os.makedirs(modelPath)
    except OSError as ee:
        pass
    print("Training the Target model for {} epoch".format(num_epoch))

    targetTrain, targetTrainLabel  = att_frame.load_data_for_trainortest(dataPath + '/targetTrain.npz')
    targetTest,  targetTestLabel   = att_frame.load_data_for_trainortest(dataPath + '/targetTest.npz')
    attackModelDataTarget, attackModelLabelsTarget, targetModelToStore, classification_y = att_frame.trainTarget(classifierType,targetTrain, targetTrainLabel, X_test=targetTest, y_test=targetTestLabel, splitData= False, inepochs=num_epoch, batch_size=100, datasetFlag = dataset)
    torch.save(targetModelToStore, modelPath + '/targetModel_{}.pkl'.format(classifierType))
    targetModelToStore = torch.load(modelPath + '/targetModel_{}.pkl'.format(classifierType))
    return targetModelToStore

def initializeShadowModelforDistill(dataset,num_epoch,dataFolderPath= './data/',modelFolderPath = './model_IncludingDistillation/',classifierType = 'cnn', num_epoch_for_distillation = 50):
    dataPath = dataFolderPath+dataset+'/PreprocessedIncludingDistillationData'
    attackerModelDataPath = dataFolderPath+dataset+'/attackerModelDataIncludingDistillation'
    modelPath = modelFolderPath + dataset
    try:
        os.makedirs(attackerModelDataPath)
    except OSError as ee:
        pass
    try:
        os.makedirs(modelPath)
    except OSError as ee:
        pass
    print("Training the Shadow model for {} epoch".format(num_epoch))
    shadowTrain, shadowTrainLabel  = att_frame.load_data_for_trainortest(dataPath + '/shadowTrain.npz')
    shadowTest,  shadowTestLabel   = att_frame.load_data_for_trainortest(dataPath + '/shadowTest.npz')
    attackModelDataShadow, attackModelLabelsShadow, shadowModelToStore, classification_y = att_frame.trainTarget(classifierType,shadowTrain, shadowTrainLabel, X_test=shadowTest, y_test=shadowTestLabel, splitData= False, inepochs=num_epoch, batch_size=100, datasetFlag = dataset) 

    torch.save(shadowModelToStore, modelPath + '/shadowModel_{}.pkl'.format(classifierType))
    shadowModelToStore = torch.load(modelPath + '/shadowModel_{}.pkl'.format(classifierType))
    return shadowModelToStore

def initializeDataIncludingDistillationData(dataset,orginialDatasetPath,dataFolderPath = './data/'):
    if(dataset == 'CIFAR10'):
        print("Loading official data from the local disk (CIFAR10)")
        dataX, dataY, testdataX, testdataY = rd.readCIFAR10(orginialDatasetPath)
        X = np.concatenate((dataX , testdataX), axis=0)
        y = np.concatenate((dataY , testdataY), axis=0)
        print("Preprocessing data, there is {} samples in the dataset".format(len(X)))
        cluster = 10000
        dataPath = dataFolderPath+dataset+'/PreprocessedIncludingDistillationData'

        toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel, distillationData, distillationDataLabel = shuffleAndSplitDataByClass_distillation(X, y, cluster)
        toTrainDataSave, toTestDataSave    = rd.preprocessingCIFAR(toTrainData, toTestData)
        shadowDataSave, shadowTestDataSave = rd.preprocessingCIFAR(shadowData, shadowTestData)
        distillationDataSave = rd.preprocessingCIFAR(distillationData, np.array([]))
    else:
        print("dataset error!")



    try:
        os.makedirs(dataPath)
    except OSError:
        pass

    np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
    np.savez(dataPath + '/targetTest.npz',  toTestDataSave, toTestLabel)
    np.savez(dataPath + '/shadowTrain.npz', shadowDataSave, shadowLabel)
    np.savez(dataPath + '/shadowTest.npz',  shadowTestDataSave, shadowTestLabel)
    np.savez(dataPath + '/distillationData.npz',  distillationDataSave, distillationDataLabel)
    
    print("Preprocessing finished\n\n")

def createAttackDataWithMetrics(dataset,dataFolderPath= './data/',modelFolderPath = './model_IncludingDistillation/',classifierType = 'cnn', TargetOrShadow='Target', batch_size=100, metricFlag='loss'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataPath = dataFolderPath+dataset+'/PreprocessedIncludingDistillationData'
    distilledmodelesPath = modelFolderPath + dataset + '/' + TargetOrShadow + '/' + classifierType
    modelPath = modelFolderPath + dataset

    attackerModelDataPath = dataFolderPath+dataset+'/attackerModelDataIncludingDistillation'
    
    datasetFlag = dataset
    print("Starting attackData creation...")
    if TargetOrShadow =='Target':
        X_train, y_train  = att_frame.load_data_for_trainortest(dataPath + '/targetTrain.npz')
        X_test, y_test   = att_frame.load_data_for_trainortest(dataPath + '/targetTest.npz')
    elif TargetOrShadow =='Shadow':
        X_train, y_train  = att_frame.load_data_for_trainortest(dataPath + '/shadowTrain.npz')
        X_test, y_test   = att_frame.load_data_for_trainortest(dataPath + '/shadowTest.npz')
    
    train_x = X_train.astype(np.float32)
    train_y = y_train.astype(np.int32)
    test_x  = X_test.astype(np.float32)
    test_y  = y_test.astype(np.int32)
    n_out = len(np.unique(train_y))
    if batch_size > len(train_y):
        batch_size = len(train_y)
    
    if(datasetFlag=='CIFAR10'):
        train_data = models.CIFARData(train_x, train_y)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loader_noShuffle = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_data = models.CIFARData(test_x, test_y)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        print("dataset error!")
    if TargetOrShadow =='Target':

        original_model = torch.load(modelPath + '/targetModel_{}.pkl'.format(classifierType)) 

        distilledModeleNames =  get_file_name(distilledmodelesPath)  
        distilledModeles = []
        for i in distilledModeleNames:
            print("loading {} from {}".format(i, distilledmodelesPath))
            distilledModeles.append(torch.load(distilledmodelesPath + '/' + i))

    elif TargetOrShadow =='Shadow':

        original_model = torch.load(modelPath + '/shadowModel_{}.pkl'.format(classifierType))

        distilledModeleNames =  get_file_name(distilledmodelesPath)
        distilledModeles = []
        for i in distilledModeleNames:
            print("loading {} from {}".format(i, distilledmodelesPath))
            distilledModeles.append(torch.load(distilledmodelesPath + '/' + i))

    attack_x, attack_y = [], []
    classification_y = []
    losses = []

    for step, (X_vector, Y_vector) in enumerate(train_loader_noShuffle):
        X_vector = X_vector.to(device)
        output = original_model(X_vector)
        out_y = output.detach().cpu()
        Y_vector_onehot = F.one_hot(Y_vector,n_out)
        
        loss_a_batch = metr.ComputeMetric(Y_vector,Y_vector_onehot,out_y,metricFlag='loss')
        loss_a_batch = loss_a_batch.squeeze()

        metrics_ori = metr.ComputeMultiMetric(Y_vector,Y_vector_onehot,out_y,metricFlag=metricFlag)
        metrics_all = metrics_ori
        for m in distilledModeles:
            output_dis = m(X_vector)
            out_y_dis = output_dis.detach().cpu()
            Y_vector_onehot = F.one_hot(Y_vector,n_out)
            metrics_dis = metr.ComputeMultiMetric(Y_vector,Y_vector_onehot,out_y_dis,metricFlag=metricFlag)
            metrics_dis = metrics_dis
            metrics_all = torch.cat([metrics_all, metrics_dis],1)


        num_metrics = metricFlag.count('&')+1
        if num_metrics == 1:
            metrics_all = metrics_all  
    
        elif num_metrics == 2:
            c,d = torch.split(metrics_all,len(Y_vector),dim=0)
            metrics_all = torch.cat([c, d],1)

        elif num_metrics == 3:
            c,d,e = torch.split(metrics_all,len(Y_vector),dim=0)
            metrics_all = torch.cat([c, d, e],1)

        elif num_metrics == 4:
            c,d,e,f = torch.split(metrics_all,len(Y_vector),dim=0)
            metrics_all = torch.cat([c, d, e, f],1)

        elif num_metrics == 5:
            c,d,e,f,g = torch.split(metrics_all,len(Y_vector),dim=0)
            metrics_all = torch.cat([c, d, e, f, g],1)
        
        attack_x.append(metrics_all)
        attack_y.append(np.ones(len(Y_vector)))
        classification_y.append(Y_vector)
        losses.append(loss_a_batch)

    for step, (X_vector, Y_vector) in enumerate(test_loader):
        X_vector = X_vector.to(device)

        output = original_model(X_vector)
        out_y = output.detach().cpu()
        Y_vector_onehot = F.one_hot(Y_vector,n_out)
        
        loss_a_batch = metr.ComputeMetric(Y_vector,Y_vector_onehot,out_y,metricFlag='loss')
        loss_a_batch = loss_a_batch.squeeze()

        metrics_ori = metr.ComputeMultiMetric(Y_vector,Y_vector_onehot,out_y,metricFlag=metricFlag)
        metrics_all = metrics_ori
        for m in distilledModeles:
            output_dis = m(X_vector)
            out_y_dis = output_dis.detach().cpu()
            Y_vector_onehot = F.one_hot(Y_vector,n_out)
            metrics_dis = metr.ComputeMultiMetric(Y_vector,Y_vector_onehot,out_y_dis,metricFlag=metricFlag)
            metrics_all = torch.cat([metrics_all, metrics_dis],1)

        num_metrics = metricFlag.count('&')+1
        if num_metrics == 1:
            metrics_all = metrics_all  
    
        elif num_metrics == 2:
            c,d = torch.split(metrics_all,len(Y_vector),dim=0)
            metrics_all = torch.cat([c, d],1)

        elif num_metrics == 3:
            c,d,e = torch.split(metrics_all,len(Y_vector),dim=0)
            metrics_all = torch.cat([c, d, e],1)

        elif num_metrics == 4:
            c,d,e,f = torch.split(metrics_all,len(Y_vector),dim=0)
            metrics_all = torch.cat([c, d, e, f],1)

        elif num_metrics == 5:
            c,d,e,f,g = torch.split(metrics_all,len(Y_vector),dim=0)
            metrics_all = torch.cat([c, d, e, f, g],1)

        attack_x.append(metrics_all)
        attack_y.append(np.zeros(len(Y_vector)))
        classification_y.append(Y_vector)
        losses.append(loss_a_batch)

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classification_y = np.concatenate(classification_y)

    losses = np.concatenate(losses)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classification_y = classification_y.astype('int32')

    losses = losses.astype('float32')
    if TargetOrShadow =='Target':
        print('AttackDataset for evaluation has {} samples'.format(len(classification_y)))
    elif TargetOrShadow =='Shadow':
        print('AttackDataset for training has {} samples'.format(len(classification_y)))

    if TargetOrShadow =='Target':
        np.savez(attackerModelDataPath + '/targetModelData_{}.npz'.format(classifierType), attack_x, attack_y, classification_y, losses)
    elif TargetOrShadow =='Shadow':
        np.savez(attackerModelDataPath + '/shadowModelData_{}.npz'.format(classifierType), attack_x, attack_y, classification_y, losses)
    


    return attack_x, attack_y, classification_y, losses


def distill_original_model(modelPath, original_model, modelType, distill_dataset, epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, datasetFlag='CIFAR10', TargetOrShadow='Target'):

    distilledModelPath_with_modelType = modelPath + '/' + modelType
    try:
        os.makedirs(distilledModelPath_with_modelType)
    except OSError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    distill_x, distill_y = distill_dataset
    n_out = len(np.unique(distill_y))
    if TargetOrShadow =='Target':
        print('Distilling target_model with {} samples, {} classes...'.format(len(distill_x), n_out))
    elif TargetOrShadow =='Shadow':
        print('Distilling shadow_model with {} samples, {} classes...'.format(len(distill_x), n_out))
    
   
    if(datasetFlag=='CIFAR10'):
        distill_data = models.CIFARData(distill_x, distill_y)
        distill_loader_noShuffle = DataLoader(distill_data, batch_size=batch_size, shuffle=False)

    else:
        print("dataset error!")

    soft_labels = []
    X = []
    classification_y=[]
    original_model = original_model.to(device)
    original_model.eval()

    for step, (X_vector, Y_vector) in enumerate(distill_loader_noShuffle):
        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)

        output = original_model(X_vector)
        out_y = output.detach().cpu()
        softmax_y = att_frame.softmax(out_y.numpy())
        
        X_vector = X_vector.detach().cpu()
        Y_vector = Y_vector.detach().cpu()

        soft_labels.append(softmax_y)
        X.append(X_vector)
        classification_y.append(Y_vector)
               
    soft_labels = np.vstack(soft_labels)
    X = np.concatenate(X)
    classification_y = np.concatenate(classification_y)

    soft_labels = soft_labels.astype('float32')
    X = X.astype('float32')
    classification_y = classification_y.astype('int32')

    if(datasetFlag=='CIFAR10'):
        #加载训练数据到loader.
        distill_data_with_softlabels = models.CIFARDataForDistill(X, classification_y, soft_labels)
        distill_loader_with_softlabels = DataLoader(distill_data_with_softlabels, batch_size=batch_size, shuffle=True)
        distill_loader_with_softlabels_noshuffle = DataLoader(distill_data_with_softlabels, batch_size=batch_size, shuffle=False)


    else:
        print("dataset error!")

    params = {}
    if(datasetFlag=='CIFAR10'):
        params['task'] = 'cifar10'
        params['input_size'] = 32
        params['num_classes'] = 10
    else:
        print("dataset error!")

    if modelType =='vgg':
        print('Using vgg model as distilled model...')
        params['conv_channels'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        params['fc_layers'] = [512, 512]
        params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        params['conv_batch_norm'] = True
        params['init_weights'] = True
        params['augment_training'] = True
        net = models.VGG(params)
        net = net.to(device)
    else:
        print("model type error!")


    net.train()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    if  modelType == 'vgg':
        l2_ratio = 0.0005
        learning_rate = 0.1
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
        momentum = 0.9
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=l2_ratio)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    else:
        print("model type error!")

    print('dataset: {},  model: {},  device: {},  epoch: {},  batch_size: {},   learning_rate: {},  l2_ratio: {}'.format(datasetFlag, modelType, device, epochs, batch_size, learning_rate, l2_ratio))
    
    count = 1
    print('Distilling...')
    for epoch in range(epochs):
        running_loss = 0

        for step, (X_vector, Y_vector, softlabel_vector) in enumerate(distill_loader_with_softlabels):
            X_vector = X_vector.to(device)
            Y_vector = Y_vector.to(device)
            softlabel_vector = softlabel_vector.to(device)
            output = net(X_vector)
            loss = criterion(output, softlabel_vector)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        torch.save(net, distilledModelPath_with_modelType +'/distilledModel_{}.pkl'.format(epoch))
        if optimizer.param_groups[0]['lr']>0.0005:
            scheduler.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, Loss: {:.5f},  lr: {}'.format(epoch + 1, running_loss, optimizer.param_groups[0]['lr']))

    print("Distilling finished!")

def DistillModel(original_model,dataset,num_epoch,dataFolderPath= './data/',modelFolderPath = './model_IncludingDistillation/',classifierType = 'cnn', TargetOrShadow='Target'):
    dataPath = dataFolderPath+dataset+'/PreprocessedIncludingDistillationData'
    modelPath = modelFolderPath + dataset + '/' + TargetOrShadow
    try:
        os.makedirs(modelPath)
    except OSError as ee:
        #print(ee)
        pass
    
    print("Distilling the {} model for {} epoch".format(TargetOrShadow, num_epoch))

    distill_x, distill_y  = att_frame.load_data_for_trainortest(dataPath + '/distillationData.npz')

    distill_dataset = (distill_x.astype(np.float32),
               distill_y.astype(np.int32))
    distill_original_model(modelPath, original_model, classifierType, distill_dataset, epochs=num_epoch, batch_size=100, learning_rate=0.001, l2_ratio=1e-7,
                       n_hidden=128, datasetFlag=dataset, TargetOrShadow=TargetOrShadow)
    
    return modelPath

def generateAttackDataForSeqMIA(dataset, classifierType, dataFolderPath ,pathToLoadData ,num_epoch ,preprocessData ,trainTargetModel ,trainShadowModel,topX=3, num_epoch_for_distillation=50,distillTargetModel=True, distillShadowModel=True, metricFlag='loss'):
    if(preprocessData):
        initializeDataIncludingDistillationData(dataset,pathToLoadData)
    if(trainTargetModel):
        targetModel = initializeTargetModelforDistill(dataset,num_epoch,classifierType =classifierType, num_epoch_for_distillation = num_epoch_for_distillation)
    else:
        modelFolderPath = './model_IncludingDistillation/'
        modelPath = modelFolderPath + dataset
        targetModel = torch.load(modelPath + '/targetModel_{}.pkl'.format(classifierType))
    if(distillTargetModel):
        distillTargetModelPath = DistillModel(targetModel, dataset, num_epoch_for_distillation, classifierType =classifierType, modelFolderPath = './model_IncludingDistillation/', TargetOrShadow='Target')
    else:
        modelFolderPath = './model_IncludingDistillation/'
        distillTargetModelPath = modelFolderPath + dataset + '/' + 'Target'
    if(trainShadowModel):
        shadowModel = initializeShadowModelforDistill(dataset,num_epoch,classifierType =classifierType, num_epoch_for_distillation = num_epoch_for_distillation)
    else:
        modelFolderPath = './model_IncludingDistillation/'
        modelPath = modelFolderPath + dataset
        shadowModel = torch.load(modelPath + '/shadowModel_{}.pkl'.format(classifierType))
    if(distillShadowModel):
        distillShadowModelPath = DistillModel(shadowModel, dataset, num_epoch_for_distillation, classifierType =classifierType, modelFolderPath = './model_IncludingDistillation/', TargetOrShadow='Shadow')
    else:
        modelFolderPath = './model_IncludingDistillation/'
        distillTargetModelPath = modelFolderPath + dataset + '/' + 'Shadow'
    targetX, targetY, target_classification_y, target_losses = createAttackDataWithMetrics(dataset,dataFolderPath= './data/',modelFolderPath = './model_IncludingDistillation/',classifierType = classifierType, TargetOrShadow='Target', batch_size=100, metricFlag = metricFlag)

    shadowX, shadowY, shadow_classification_y, shadow_losses  = createAttackDataWithMetrics(dataset,dataFolderPath= './data/',modelFolderPath = './model_IncludingDistillation/',classifierType = classifierType, TargetOrShadow='Shadow', batch_size=100, metricFlag = metricFlag)
    
    return targetX, targetY, shadowX, shadowY, target_classification_y, shadow_classification_y, target_losses, shadow_losses








