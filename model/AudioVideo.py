from utils import *
from os import path
from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
from .Resnet import resnet18


class AudioEncoder(nn.Module):
    def __init__(self, config=None):
        super(AudioEncoder, self).__init__()
        self.audio_net = resnet18(modality='audio')

    def forward(self, audio):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        return a

class VideoEncoder(nn.Module):
    def __init__(self, config=None, fps=1):
        super(VideoEncoder, self).__init__()
        self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video):
        v = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v

class AVGBShareClassifier(nn.Module):
    def __init__(self, config):
        super(AVGBShareClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config, config['fps'])
        self.hidden_dim = 512

        self.embedding_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.embedding_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.num_class = config['setting']['num_class']

        self.fc_out = nn.Linear(256, self.num_class)
        self.additional_layers_a = nn.ModuleList()
        self.additional_layers_v = nn.ModuleList()
        self.relu = nn.ReLU()

    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        return a_feature, v_feature

    def add_layer(self, is_a=True):
        new_layer = nn.Linear(self.hidden_dim, 256).cuda()
        nn.init.xavier_normal_(new_layer.weight)
        nn.init.constant_(new_layer.bias, 0)
        if is_a:
            self.additional_layers_a.append(new_layer)
        else:
            self.additional_layers_v.append(new_layer)
    def classfier(self, x, is_a=True):
        if is_a:
            result_a = self.embedding_a(x)
            feature = self.fc_out(result_a)
            o_fea = feature
            add_fea = None
            i = 0
            layerlen = len(self.additional_layers_a)
            for layer in self.additional_layers_a:
                addf = self.relu(layer(x))
                add_fea = self.fc_out(addf)
                feature = feature + add_fea
                i=i+1
                if i < layerlen:
                    o_fea = feature
        else:
            result_v = self.embedding_v(x)
            feature = self.fc_out(result_v)
            o_fea = feature
            add_fea = None
            j = 0
            layerlen = len(self.additional_layers_v)
            for layer in self.additional_layers_v:
                addf = self.relu(layer(x))
                add_fea = self.fc_out(addf)
                feature = feature + add_fea
                j=j+1
                if j < layerlen:
                    o_fea = feature
        return feature, o_fea, add_fea