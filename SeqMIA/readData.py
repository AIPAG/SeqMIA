import pickle
from tkinter import Y
import numpy as np
import os
from PIL import Image

import pandas as pd


def readCIFAR10(data_path):
    print(data_path)
    for i in range(5):
        f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
        train_data_dict = pickle.load(f, encoding='iso-8859-1')
        f.close()
        if i == 0:
            X = train_data_dict["data"]
            y = train_data_dict["labels"]
            continue
        X = np.concatenate((X, train_data_dict["data"]), axis=0)
        y = np.concatenate((y, train_data_dict["labels"]), axis=0)

    f = open(data_path + '/test_batch', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()
    XTest = np.array(test_data_dict["data"])
    yTest = np.array(test_data_dict["labels"])

    return X, y, XTest, yTest


def reshape_for_save(raw_data):
    raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
    raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3))
    raw_data = raw_data.transpose(0, 3, 1, 2)
    return raw_data.astype(np.float32)


def rescale(raw_data, offset, scale):
    newdata = reshape_for_save(raw_data)
    return (newdata - offset) / scale


def preprocessingCIFAR(toTrainData, toTestData):
    if (toTestData.size != 0):
        print("train data size:")
        print(np.shape(toTrainData))
        print("test data size:")
        print(np.shape(toTestData))

        newdata = reshape_for_save(toTrainData)
        offset = np.mean(newdata,
                         0)

        scale = np.std(newdata, 0).clip(
            min=1)
        return rescale(toTrainData, offset, scale), rescale(toTestData, offset, scale)
    else:
        print("distillation data size:")
        print(np.shape(toTrainData))

        newdata = reshape_for_save(toTrainData)
        offset = np.mean(newdata,
                         0)

        scale = np.std(newdata, 0).clip(
            min=1)
        return rescale(toTrainData, offset, scale)
