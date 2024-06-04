# from models.MVCNN import MVCNN
from __future__ import division, absolute_import
from models.resnet import resnet18
from tools.utils import calculate_accuracy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()

        ## global section
        resnet18 = models.resnet18(pretrained=True)
        resnet18 = list(resnet18.children())[:-1]
        self.img_global_net = nn.Sequential(*resnet18)
        self.linear1 = nn.Linear(512, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ) # alexNet的 network architecture
        self.dimension = 128
        self.one_by_one_conv = nn.Conv2d(512,self.dimension,1)
        self.fc1 = nn.Linear(self.dimension**2,1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.4)

    def sign_sqrt(self, x):
        x = torch.mul(torch.sign(x),torch.sqrt(torch.abs(x)+1e-12))
        return x

    def bilinear_pooling(self,x):
        return torch.mm(x,x.t())

    def forward(self,views):
        
        views = views.transpose(0,1)
        view_pool = []
        for v in views:
            v = self.img_global_net(v)
            v = self.one_by_one_conv(v)
            v = self.sign_sqrt(v)
            view_pool.append(v)

        y = torch.stack(view_pool)
        local_feature = y.transpose(0,1) #[batch_size,view_num,dimension,width,height]

        res = []
        # each memeber in batch
        for b in local_feature:
            b = b.transpose(0,1) # [dimension of local feature,views,width,height]
            b = b.contiguous()
            b = b.view(b.size(0),-1) # [dimension of local feature, views*w*h]
            b = self.bilinear_pooling(b) # bilinear pooling
            b = b.view(-1) # before Late sqrt, reshape the 2D matrix into a 1D vector
            b = self.sign_sqrt(b) # late sqrt layer 这个特征是关键的
            b = b / (torch.norm(b,2)+1e-8)
            res.append(b)

        img_local_feature = torch.stack(res) # [batch_size, num_local_features**2]
        img_local_feature = self.drop1(F.relu(self.bn1(self.fc1(img_local_feature))))
        img_local_feature = self.drop2(F.relu(self.bn2(self.fc2(img_local_feature))))
        return img_local_feature


def mhbnn(pretrained=None):
    model = ImageNet()
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict) # 更新已有的值
    return model

class CorrNet(nn.Module):
    
    def __init__(self,img_net,pt_net, mesh_net,num_classes):
        super(CorrNet, self).__init__()
        self.img_net = img_net
        self.pt_net = pt_net
        self.mesh_net = mesh_net
        self.num_classes=num_classes
	#shared head for all feature encoders
        self.head = nn.Sequential(*[nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.num_classes)])
        #self.head_img = nn.Sequential(*[nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.num_classes)])

    def forward(self, pt, views, centers, corners, normals, neighbor_index):

	#extract image features
        img_feat = self.img_net(views)
        #extract pt features
        pt_feat = self.pt_net(pt)
	#extract mesh features
        mesh_feat = self.mesh_net(centers, corners, normals, neighbor_index)

        #cmb_feat = (img_feat  + mesh_feat)/2.0

	#the classification predictions based on image features
        img_pred = self.head(img_feat)

	#the classification prediction based on mesh featrues
        mesh_pred = self.head(mesh_feat)
        #the classification prediction based on pt featrues
        pt_pred = self.head(pt_feat)

        # cmb_pred = self.head(cmb_feat)

        return img_pred, pt_pred,mesh_pred, img_feat, pt_feat,mesh_feat

