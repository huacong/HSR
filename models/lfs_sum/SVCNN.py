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
        self.drop = nn.Dropout(0.4)

    def forward(self,views):
        views = views.transpose(0,1)
        final_feat = None
        for v in views:
            img_feat = self.img_global_net(v)
            img_feat = img_feat.squeeze(3)
            img_feat = img_feat.squeeze(2)
            if final_feat == None:
                final_feat = img_feat
            else:
                final_feat = final_feat + img_feat
        final_feat = self.drop(F.relu(self.bn6(self.linear1(final_feat))))
        return final_feat

def mhbnn(pretrained=None):
    model = ImageNet()
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict) #
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

