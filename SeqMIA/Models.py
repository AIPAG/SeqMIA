from re import X
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import math

import torch.nn.utils.rnn as rnn_utils


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class VGG(nn.Module):
    def __init__(self, params):
        super(VGG, self).__init__()

        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels']
        self.fc_layer_sizes = params['fc_layers']

        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.init_weights = params['init_weights']
        self.augment_training = params['augment_training']
        self.num_output = 1

        self.init_conv = nn.Sequential()

        self.layers = nn.ModuleList()
        input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel

        fc_input_size = cur_input_size * cur_input_size * self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class TrData(Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]


def collate_fn(trs):
    onetr = trs[0]
    onepoint_size = onetr.size(1)
    input_size = onepoint_size - 1
    trs.sort(key=lambda x: len(x), reverse=True)
    tr_lengths = [len(sq) for sq in trs]
    trs = rnn_utils.pad_sequence(trs, batch_first=True, padding_value=0)
    var_x = trs[:, :, 1:input_size + 1]
    tmpy = trs[:, :, 0]
    var_y = tmpy[:, 0]
    return var_x, var_y, tr_lengths


class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, num_classes=2, batch_size=1):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, len_of_oneTr):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        batch_x_pack = rnn_utils.pack_padded_sequence(x,
                                                      len_of_oneTr, batch_first=True).cuda()
        out, (h1, c1) = self.layer1(batch_x_pack, (h0, c0))
        out = self.layer2(h1)
        return out, h1


class LSTM_Attention(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, num_classes=2, batch_size=1):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer3 = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, len_of_oneTr):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        batch_x_pack = rnn_utils.pack_padded_sequence(x,
                                                      len_of_oneTr, batch_first=True).cuda()
        out, (h1, c1) = self.layer1(batch_x_pack, (h0, c0))
        outputs, lengths = rnn_utils.pad_packed_sequence(out, batch_first=True)
        permute_outputs = outputs.permute(1, 0, 2)
        atten_energies = torch.sum(h1 * permute_outputs,
                                   dim=2)

        atten_energies = atten_energies.t()

        scores = F.softmax(atten_energies, dim=1)

        scores = scores.unsqueeze(0)

        permute_permute_outputs = permute_outputs.permute(2, 1, 0)
        context_vector = torch.sum(scores * permute_permute_outputs,
                                   dim=2)
        context_vector = context_vector.t()
        context_vector = context_vector.unsqueeze(0)
        out2 = torch.cat((h1, context_vector), 2)
        out = self.layer3(out2)
        return out, out2


class CIFARData(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        img = self.X_train[idx]
        label = self.y_train[idx]
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label


class CIFARDataForDistill(Dataset):
    def __init__(self, X_train, y_train, softlabel):
        self.X_train = X_train
        self.y_train = y_train
        self.softlabel = softlabel

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        img = self.X_train[idx]
        label = self.y_train[idx]
        softlabel = self.softlabel[idx]
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label, softlabel
