from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F, ParameterList, ModuleList, Parameter
from torch.autograd import Variable
import torchvision
import numpy as np

from models import resnet3d

class AttrClassifierLinear(nn.Module):
    def __init__(self, feature_dim, attr_len):
        super(AttrClassifierLinear, self).__init__()
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, attr_len)
    def forward(self, x):
        return self.classifier(x)


class AttrClassifierHead(nn.Module):
    def __init__(self, feature_dim, attr_len):
        super(AttrClassifierHead, self).__init__()
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, attr_len)
    def forward(self, x, b, t):
        # x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = torch.mean(x, 1)
        # f = x.view(b, self.feature_dim)
        return self.classifier(x)

class AttrClassifierHeadWithAttention(nn.Module):
    def __init__(self, feature_dim, attr_len):
        super(AttrClassifierHeadWithAttention, self).__init__()
        self.feature_dim = feature_dim
        self.middle_dim = 256 # middle layer dimension
        self.attention_conv = nn.Conv1d(self.feature_dim, self.middle_dim, 1)
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.classifier = nn.Linear(feature_dim, attr_len)
    def forward(self, x, b, t):
        a = x.permute(0, 2, 1)
        a = F.relu(self.attention_conv(a))
        # a = F.avg_pool2d(a, a.size()[-2:])
        # a = a.view(b, t, self.middle_dim)
        # a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        t_atten = F.softmax(a, dim=1)
        t_atten = torch.unsqueeze(t_atten, -1)
        t_atten = t_atten.expand(b, t, self.feature_dim)
        # x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        att_x = torch.mul(x, t_atten)
        att_x = torch.sum(att_x, 1)
        f = att_x.view(b, self.feature_dim)
        return F.softmax(a, dim=1), self.classifier(f)


class MultiLabelLinearAttributeModule(nn.Module):
    def __init__(self, feature_dim, attr_lens):
        super(MultiLabelLinearAttributeModule, self).__init__()
        self.feature_dim = feature_dim
        self.attr_lens = attr_lens
        self.make_fcs()

    def make_fcs(self):
        self.classifiers = []
        for l in self.attr_lens:
            self.classifiers.append(AttrClassifierLinear(self.feature_dim, l))
        self.classifiers = ModuleList(self.classifiers)

    def forward(self, x):
        # out = x.view(x.size(0), self.feature_dim)
        out_labels = []
        for c in self.classifiers:
            out = c(x)
            out_labels.append(out)
        return out_labels

class MultiLabelAttributeModule(nn.Module):
    def __init__(self, feature_dim, attr_lens):
        super(MultiLabelAttributeModule, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)
        self.attr_lens = attr_lens
        self.relu = nn.ReLU()
        self.make_fcs()

    def make_fcs(self):
        self.classifiers = []
        for l in self.attr_lens:
            self.classifiers.append(AttrClassifierHead(self.feature_dim, l))
        self.classifiers = ModuleList(self.classifiers)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        c = x.size(2)
        w = x.size(3)
        h = x.size(4)
        out = x.view(b * t, c, w, h)
        out = self.conv1(out)
        temp_f = self.relu(out)
        temp_f = F.avg_pool2d(temp_f, temp_f.size()[2:])
        temp_f = temp_f.view(b, t, -1)
        out_labels = []
        for c in self.classifiers:
            out = c(temp_f, b, t)
            out_labels.append(out)
        return out_labels

class MultiLabelAttributeModuleWithTAttention(nn.Module):
    def __init__(self, feature_dim, attr_lens):
        super(MultiLabelAttributeModuleWithTAttention, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)
        self.attr_lens = attr_lens
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=0.5)
        self.make_fcs()

    def make_fcs(self):
        self.classifiers = []
        for l in self.attr_lens:
            self.classifiers.append(AttrClassifierHeadWithAttention(self.feature_dim, l))
        self.classifiers = ModuleList(self.classifiers)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        c = x.size(2)
        w = x.size(3)
        h = x.size(4)
        out = x.view(b * t, c, w, h)
        out = self.conv1(out)
        temp_f = self.relu(out)
        temp_f = F.avg_pool2d(temp_f, temp_f.size()[2:])
        temp_f = temp_f.view(b, t, -1)
        out_labels = []
        attentions = []
        temp_f_dp = self.dp(temp_f)
        for c in self.classifiers:
            a, out = c(temp_f_dp, b, t)
            attentions.append(a)
            out_labels.append(out)
        return out_labels
class MultiLabelAttributeModuleRNN(nn.Module):
    def __init__(self, feature_dim, attr_lens):
        super(MultiLabelAttributeModuleRNN, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)
        self.attr_lens = attr_lens
        self.relu = nn.ReLU()
        self.make_fcs()
        self.hidden_dim = 2048
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

    def make_fcs(self):
        self.classifiers = []
        for l in self.attr_lens:
            self.classifiers.append(AttrClassifierHeadWithAttention(self.feature_dim, l))
        self.classifiers = ModuleList(self.classifiers)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        c = x.size(2)
        w = x.size(3)
        h = x.size(4)
        out = x.view(b * t, c, w, h)
        out = self.conv1(out)
        temp_f = self.relu(out)
        temp_f = F.avg_pool2d(temp_f, temp_f.size()[2:])
        temp_f = temp_f.view(b, t, -1)
        temp_f_rnn, (h_n, c_n) = self.lstm(temp_f)
        out_labels = []
        temp_f = temp_f + temp_f_rnn
        for c in self.classifiers:
            out = c(temp_f, b, t)
            out_labels.append(out)
        return out_labels

class AttrResNet50TPBaseline(nn.Module):
    def __init__(self, attr_lens, model_type="tp", **kwargs):
        super(AttrResNet50TPBaseline, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feature_dim = 2048
        self.idrelated_classifier = MultiLabelLinearAttributeModule(self.feature_dim, attr_lens[0])
        self.idunrelated_classifier = MultiLabelLinearAttributeModule(self.feature_dim, attr_lens[1])

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        result = []
        # x = x.view(b, t, -1)
        x = torch.mean(x, 1)
        id_related_labels = self.idrelated_classifier(x)
        id_unrelated_labels = self.idunrelated_classifier(x)
        result.extend(id_related_labels)
        result.extend(id_unrelated_labels)
        return result

class AttrResNet50TP(nn.Module):
    def __init__(self, attr_lens, model_type="tp", **kwargs):
        super(AttrResNet50TP, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feature_dim = 2048
        if model_type == 'tp':
            self.idrelated_classifier = MultiLabelAttributeModule(self.feature_dim, attr_lens[0])
            self.idunrelated_classifier = MultiLabelAttributeModule(self.feature_dim, attr_lens[1])
        elif model_type == 'ta':
            self.idrelated_classifier = MultiLabelAttributeModuleWithTAttention(self.feature_dim, attr_lens[0])
            self.idunrelated_classifier = MultiLabelAttributeModuleWithTAttention(self.feature_dim, attr_lens[1])
        elif model_type == 'rnn':
            self.idrelated_classifier = MultiLabelAttributeModuleRNN(self.feature_dim, attr_lens[0])
            self.idunrelated_classifier = MultiLabelAttributeModuleRNN(self.feature_dim, attr_lens[1])

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = x.view(b, t, self.feature_dim, x.size(2), x.size(3))
        result = []
        id_related_labels = self.idrelated_classifier(x)
        id_unrelated_labels = self.idunrelated_classifier(x)
        result.extend(id_related_labels)
        result.extend(id_unrelated_labels)
        return result

class AttrResNet503D(nn.Module):
    def __init__(self, attr_lens, sample_width, sample_height, sample_duration, model_type="tp", **kwargs):
        super(AttrResNet503D, self).__init__()
        resnet50 = resnet3d.resnet50(sample_width=sample_width, sample_height=sample_height, sample_duration=sample_duration)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feature_dim = 512
        self.idrelated_classifier = MultiLabelLinearAttributeModule(self.feature_dim, attr_lens[0])
        self.idunrelated_classifier = MultiLabelLinearAttributeModule(self.feature_dim, attr_lens[1])

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x=x.permute(0,2,1,3,4)
        # x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = x.view(b, self.feature_dim, x.size(3), x.size(4))
        x = F.avg_pool2d(x, x.size()[2:])
        result = []
        x = x.view(b, self.feature_dim)
        id_related_labels = self.idrelated_classifier(x)
        id_unrelated_labels = self.idunrelated_classifier(x)
        result.extend(id_related_labels)
        result.extend(id_unrelated_labels)
        return result

class AttrResNet50RNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(AttrResNet50RNN, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feature_dim = 2048
        # self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        output, (h_n, c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)
        f = f.view(b, self.hidden_dim)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))