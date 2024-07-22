from re import X
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset  # 导入抽象类Dataset
import numpy as np
import torch.nn.functional as F
import attackMethodsFramework as att_frame


def ComputeMetric(labels, labels_onehot_softlabels, pre_vectors, metricFlag):
    if metricFlag == 'loss':
        metrics = -torch.sum(labels_onehot_softlabels * F.log_softmax(pre_vectors, dim=1), dim=1)
        metrics = metrics.unsqueeze(1)

    if metricFlag == 'max':
        pre_vectors = pre_vectors.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)
        metrics = att_frame.clipDataTopX(pre_vectors, top=1)
        metrics = torch.from_numpy(metrics)

    if metricFlag == "mentropy":
        pre_vectors = pre_vectors.numpy()
        labels = labels.numpy()

        pre_vectors = att_frame.softmax(pre_vectors)
        neg_log_probs = -np.log(np.maximum(pre_vectors, 1e-30))
        reverse_probs = 1 - pre_vectors
        neg_log_reverse_probs = -np.log(np.maximum(reverse_probs, 1e-30))
        modified_probs = np.copy(pre_vectors)
        modified_probs[range(labels.size), labels] = reverse_probs[range(labels.size), labels]
        modified_log_probs = np.copy(neg_log_reverse_probs)
        modified_log_probs[range(labels.size), labels] = neg_log_probs[range(labels.size), labels]
        metrics = np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
        metrics = torch.from_numpy(metrics)
        metrics = metrics.unsqueeze(1)

    if metricFlag == "sd":
        pre_vectors = pre_vectors.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)
        metrics = np.std(pre_vectors, axis=1)
        metrics = torch.from_numpy(metrics)
        metrics = metrics.unsqueeze(1)

    if metricFlag == "entropy":
        pre_vectors = pre_vectors.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)
        negative_logs = -np.log(np.maximum(pre_vectors, 1e-30))
        metrics = np.sum(np.multiply(pre_vectors, negative_logs),
                         axis=1)
        metrics = torch.from_numpy(metrics)
        metrics = metrics.unsqueeze(1)

    return metrics


def ComputeMultiMetric(labels, labels_onehot_softlabels, pre_vectors,
                       metricFlag):
    if metricFlag == 'loss&entropy':
        metrics1 = -torch.sum(labels_onehot_softlabels * F.log_softmax(pre_vectors, dim=1), dim=1)
        metrics1 = metrics1.unsqueeze(1)
        pre_vectors = pre_vectors.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)
        negative_logs = -np.log(np.maximum(pre_vectors, 1e-30))
        metrics2 = np.sum(np.multiply(pre_vectors, negative_logs),
                          axis=1)
        metrics2 = torch.from_numpy(metrics2)
        metrics2 = metrics2.unsqueeze(1)

        metrics = torch.cat([metrics1, metrics2], 0)

    if metricFlag == 'loss&mentropy':
        metrics1 = -torch.sum(labels_onehot_softlabels * F.log_softmax(pre_vectors, dim=1), dim=1)
        metrics1 = metrics1.unsqueeze(1)
        pre_vectors = pre_vectors.numpy()
        labels = labels.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)
        neg_log_probs = -np.log(np.maximum(pre_vectors, 1e-30))
        reverse_probs = 1 - pre_vectors
        neg_log_reverse_probs = -np.log(np.maximum(reverse_probs, 1e-30))
        modified_probs = np.copy(pre_vectors)
        modified_probs[range(labels.size), labels] = reverse_probs[range(labels.size), labels]
        modified_log_probs = np.copy(neg_log_reverse_probs)
        modified_log_probs[range(labels.size), labels] = neg_log_probs[range(labels.size), labels]
        metrics2 = np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
        metrics2 = torch.from_numpy(metrics2)
        metrics2 = metrics2.unsqueeze(1)

        metrics = torch.cat([metrics1, metrics2], 0)

    if metricFlag == 'loss&max&sd&entropy&mentropy':
        loss_metrics = -torch.sum(labels_onehot_softlabels * F.log_softmax(pre_vectors, dim=1), dim=1)
        loss_metrics = loss_metrics.unsqueeze(1)

        pre_vectors = pre_vectors.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)

        # max
        max_metrics = att_frame.clipDataTopX(pre_vectors, top=1)
        max_metrics = torch.from_numpy(max_metrics)

        # sd
        sd_metrics = np.std(pre_vectors, axis=1)
        sd_metrics = torch.from_numpy(sd_metrics)
        sd_metrics = sd_metrics.unsqueeze(1)
        negative_logs = -np.log(np.maximum(pre_vectors, 1e-30))
        entropy_metrics = np.sum(np.multiply(pre_vectors, negative_logs),
                                 axis=1)
        entropy_metrics = torch.from_numpy(entropy_metrics)
        entropy_metrics = entropy_metrics.unsqueeze(1)
        labels = labels.numpy()
        neg_log_probs = -np.log(np.maximum(pre_vectors, 1e-30))
        reverse_probs = 1 - pre_vectors
        neg_log_reverse_probs = -np.log(np.maximum(reverse_probs, 1e-30))
        modified_probs = np.copy(pre_vectors)
        modified_probs[range(labels.size), labels] = reverse_probs[range(labels.size), labels]
        modified_log_probs = np.copy(neg_log_reverse_probs)
        modified_log_probs[range(labels.size), labels] = neg_log_probs[range(labels.size), labels]
        mentropy_metrics = np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
        mentropy_metrics = torch.from_numpy(mentropy_metrics)
        mentropy_metrics = mentropy_metrics.unsqueeze(1)
        metrics = torch.cat([loss_metrics, max_metrics, sd_metrics, entropy_metrics, mentropy_metrics],
                            0)
    if metricFlag == 'loss':
        metrics = -torch.sum(labels_onehot_softlabels * F.log_softmax(pre_vectors, dim=1), dim=1)
        metrics = metrics.unsqueeze(1)

    if metricFlag == 'max':
        pre_vectors = pre_vectors.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)
        metrics = att_frame.clipDataTopX(pre_vectors, top=1)
        metrics = torch.from_numpy(metrics)

    if metricFlag == "mentropy":
        pre_vectors = pre_vectors.numpy()
        labels = labels.numpy()

        pre_vectors = att_frame.softmax(pre_vectors)
        neg_log_probs = -np.log(np.maximum(pre_vectors, 1e-30))
        reverse_probs = 1 - pre_vectors
        neg_log_reverse_probs = -np.log(np.maximum(reverse_probs, 1e-30))
        modified_probs = np.copy(pre_vectors)
        modified_probs[range(labels.size), labels] = reverse_probs[range(labels.size), labels]
        modified_log_probs = np.copy(neg_log_reverse_probs)
        modified_log_probs[range(labels.size), labels] = neg_log_probs[range(labels.size), labels]
        metrics = np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
        metrics = torch.from_numpy(metrics)
        metrics = metrics.unsqueeze(1)

    if metricFlag == "sd":
        pre_vectors = pre_vectors.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)
        metrics = np.std(pre_vectors, axis=1)
        metrics = torch.from_numpy(metrics)
        metrics = metrics.unsqueeze(1)

    if metricFlag == "entropy":
        pre_vectors = pre_vectors.numpy()
        pre_vectors = att_frame.softmax(pre_vectors)
        negative_logs = -np.log(np.maximum(pre_vectors, 1e-30))
        metrics = np.sum(np.multiply(pre_vectors, negative_logs),
                         axis=1)
        metrics = torch.from_numpy(metrics)
        metrics = metrics.unsqueeze(1)
    return metrics
