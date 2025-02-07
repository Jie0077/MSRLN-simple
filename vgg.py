#coding=utf-8
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from kymatio.torch import Scattering2D###
#from s2d import Scattering2D###
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'



scattering = Scattering2D(J=2, shape=(32, 32))
scattering = scattering.cuda()

######the other layers

cfg =  [ 512, 512, 512, 'M', 512, 512, 512, 'M']




class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features0 = self._make_layers0()
        self.features1 = self._make_layers1()
        self.features2 = self._make_layers2()
        self.features3 = self._make_layers3()
        self.features4 = self._make_layers4(cfg)
        self.classifier = nn.Linear(512, 10)
        self.K = 16*81
        self.bn = nn.BatchNorm2d(self.K)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        out = self.features0(x)
        out_tmp = out
        out = self.maxpool(out)
        tmp = out.reshape(out.size(0), 16, 32, 32)
        scatter_feature = scattering(tmp)
        scatter_feature = self.bn(scatter_feature.view(out.size(0), self.K, 8, 8))
        scatter_feature = scatter_feature.reshape(out.size(0), 81, 32, 32)
        out = out_tmp + scatter_feature[:, 17:81, :, :]*0.01
        out = out.detach()    
        out = self.features1(out)
        out = self.features2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.features3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.features4(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


    def _make_layers0(self):
        layers = []
        in_channels = 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64)]
        return nn.Sequential(*layers)


    def _make_layers1(self):
        layers = []
        in_channels = 64
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _make_layers2(self):
        layers = []
        in_channels = 64
        layers += [nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   nn.ReLU(inplace=True)]
        in_channels = 128
        layers += [nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128)]
        return nn.Sequential(*layers)

    def _make_layers3(self):
        layers = []
        in_channels = 128
        layers += [nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)]
        in_channels = 256
        layers += [nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)]
        in_channels = 256
        layers += [nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256)]
        return nn.Sequential(*layers)

    def _make_layers4(self, cfg):
        layers = []
        in_channels = 256
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

       


  

