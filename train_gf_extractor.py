from __future__ import division, absolute_import
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
import json
import logging
import scipy
from models.dgcnn import DGCNN
from models.meshnet import MeshNet
from models.SVCNN import SingleViewNet,CorrNet
from models.MVCNN import MVCNN
from tools.test_dataloader import TestDataloader
from tools.triplet_dataloader import TripletDataloader
from tools.utils import calculate_accuracy
from center_loss import CrossModalCenterLoss
from triplet_center_loss import TripletCenterLoss
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from util.append_feature import append_feature
import warnings
from util.OS_utils import split_trainval,res2tab,acc_score, map_score
from pathlib import Path

warnings.filterwarnings('ignore',category=FutureWarning)

# Create the log
logger = logging.getLogger("Cross-Modal Model retrieval")
logger.setLevel(logging.INFO)
device = torch.device("cuda")
def log_string(str):
    logger.info(str)
    print(str)

def calc_map_label(source, target, label_test, name):
    source = normalize(source, norm='l1', axis=1)
    target = normalize(target, norm='l1', axis=1)
    dist = cdist(source, target, 'cosine')
    ord = dist.argsort()
    num = dist.shape[0]
    res = []
    for i in range(num):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(num):
            if label_test[i] == label_test[order[j]]:  # 你这个不用计算距离 也能得出来啊
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]  # p/r 这是一个物体的AP值
        else:
            res += [0]
    mAP = np.mean(res)
    mAP_round = round(mAP * 100, 2)
    log_string("%s mAP:%s" % (name, str(mAP_round)))

def test(img_net,mesh_net,pt_net,test_data_loader,test_set):
    
    img_net = img_net.eval()
    mesh_net = mesh_net.eval()
    pt_net = pt_net.eval()

    log_string('-----------------------------------test---------------------------------')
    iteration = 0
    point_correct = 0.0
    mulview_correct = 0.0
    mesh_correct = 0.0
    batch_id = 0    
    img_feature_set = None
    mesh_feature_set = None
    point_feature_set = None
    label_feature_set = None
    for data in test_data_loader:
        print("batch: %d/%d" % (batch_id,len(test_data_loader)))
        pt, img_list, centers, corners, normals, neighbor_index, target = data
        img_v1,img_v2,img_v3,img_v4 = img_list
        views = np.stack(img_list,axis=1)
        views = torch.from_numpy(views).to('cuda')
        
        img_v1 = Variable(img_v1).to('cuda')
        img_v2 = Variable(img_v2).to('cuda')
        img_v3 = Variable(img_v3).to('cuda')
        img_v4 = Variable(img_v4).to('cuda')

        pt = Variable(pt).to('cuda')
        pt = pt.permute(0,2,1)
        
        print("pre:",target.shape)
        
        target = target[:,0]
        # torch.Size([96, 1])
        target = Variable(target).to('cuda')

        print(pt.shape,target.shape)

        
        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
        neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))


        img_feat = 0.5*(img_net(img_v1,img_v2)+img_net(img_v3,img_v4))
        pt_feat = pt_net(pt)
        mesh_feat = mesh_net(centers, corners, normals, neighbor_index)

        
        ### append feature
        img_feature_set = append_feature(img_feature_set,img_feat.cpu().data.numpy())
        mesh_feature_set = append_feature(mesh_feature_set, mesh_feat.cpu().data.numpy())
        point_feature_set = append_feature(point_feature_set,pt_feat.cpu().data.numpy())
        label_feature_set = append_feature(label_feature_set,target.cpu().data.numpy(),flatten = True)


        iteration = iteration + 1
        batch_id = batch_id + 1

    calc_map_label(img_feature_set,img_feature_set,label_feature_set,"img to img")
    calc_map_label(img_feature_set,mesh_feature_set,label_feature_set,"img to mesh")
    calc_map_label(img_feature_set,point_feature_set,label_feature_set,"img to point")
    calc_map_label(mesh_feature_set,mesh_feature_set,label_feature_set,"Mesh to mesh")
    calc_map_label(mesh_feature_set,img_feature_set,label_feature_set,"Mesh to img")
    calc_map_label(mesh_feature_set,point_feature_set,label_feature_set,"Mesh to Point")
    calc_map_label(point_feature_set,point_feature_set,label_feature_set,"point to point")
    calc_map_label(point_feature_set,img_feature_set,label_feature_set,"point to img")
    calc_map_label(point_feature_set,mesh_feature_set,label_feature_set,"point to mesh")

    

def training(args):
    
    file_handler = logging.FileHandler('%s/%s.txt' % ('./logs', args.log)) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER....')

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    img_net = SingleViewNet()
    pt_net = DGCNN()
    mesh_net = MeshNet()
    model = CorrNet(img_net, pt_net, mesh_net, num_classes=args.num_classes)
    model.train(True)
    model = model.to('cuda')
    model = torch.nn.DataParallel(model)
    #cross entropy loss for classification
    ce_criterion = nn.CrossEntropyLoss()
    cmc_criterion = CrossModalCenterLoss(num_classes=args.num_classes, feat_dim=512, use_gpu=True)
    #mse loss
    mse_criterion = nn.MSELoss()
 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_centloss = optim.SGD(cmc_criterion.parameters(), lr=args.lr_center)

    train_set = TripletDataloader(dataset = args.dataset, num_points = args.num_points, num_classes=args.num_classes, dataset_dir=args.dataset_dir,  partition='train')
    test_set = TestDataloader(dataset=args.dataset, num_points = args.num_points , dataset_dir = args.dataset_dir, partition= 'test')
    train_data_loader_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=8)
    test_data_loader_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=False, num_workers=8)
    iteration = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        point_correct = 0.0
        img_correct = 0.0
        mesh_correct = 0.0
        batch_id = 0
        img_net.train(True)
        pt_net.train(True)
        mesh_net.train(True)
        model.train(True)
        for data in train_data_loader_loader:
        
            print("epoch:%d [%d/%d]"%(epoch, batch_id, len(train_data_loader_loader)))
            pt, img_list, centers, corners, normals, neighbor_index, target, target_vec = data
            img_v1,img_v2,img_v3,img_v4 = img_list
            img_v1 = Variable(img_v1).to('cuda')
            img_v2 = Variable(img_v2).to('cuda')
            img_v3 = Variable(img_v3).to('cuda')
            img_v4 = Variable(img_v4).to('cuda')
            pt = Variable(pt).to('cuda')
            pt = pt.permute(0,2,1)
            target = target[:,0]
            target = Variable(target).to('cuda')
            # target_vec = Variable(target_vec).to('cuda')
            #print("target1.shape",target.shape,target_vec.shape)
            centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
            corners = Variable(torch.cuda.FloatTensor(corners.cuda())) 
            normals = Variable(torch.cuda.FloatTensor(normals.cuda())) 
            neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda())) 

            optimizer.zero_grad()
            optimizer_centloss.zero_grad()

            img_pred, pt_pred, mesh_pred, img_feat, pt_feat, mesh_feat = model(pt, img_v1, img_v2, centers, corners, normals, neighbor_index)

            #cross-entropy loss for all the three modalities
            #print("pt_pred:",pt_pred.shape,"**** target:",target.shape)
            pt_ce_loss = ce_criterion(pt_pred, target)
            print("pt_ce_loss:",pt_ce_loss.item())
            img_ce_loss = ce_criterion(img_pred, target)
            print("img_ce_loss:",img_ce_loss.item())
            mesh_ce_loss = ce_criterion(mesh_pred, target)
            print("mesh_ce_loss:",mesh_ce_loss.item())     
            ce_loss = pt_ce_loss + img_ce_loss + mesh_ce_loss
            #cross-modal center loss 
            cmc_loss = cmc_criterion(torch.cat((img_feat, pt_feat, mesh_feat), dim = 0), torch.cat((target, target, target), dim = 0))

            # MSE Loss   
            img_pt_mse_loss = mse_criterion(img_feat, pt_feat)
            img_mesh_mse_loss = mse_criterion(img_feat, mesh_feat)
            mesh_pt_mse_loss = mse_criterion(mesh_feat, pt_feat)

            mse_loss = img_pt_mse_loss + img_mesh_mse_loss + mesh_pt_mse_loss 
            
	        #weighted the three losses as final loss 
            loss = 10*ce_loss + args.weight_center * cmc_loss +  0.1 * mse_loss
            
            loss.backward()
            optimizer.step()
           
            for param in cmc_criterion.parameters():
                param.grad.data *= (1. / args.weight_center)

            optimizer_centloss.step()

            _, pt_pred = torch.max(pt_pred, dim=1)
            _, img_pred = torch.max(img_pred, dim=1)
            _, mh_pred = torch.max(mesh_pred, dim=1)

            point_correct += torch.sum(pt_pred == target.data)
            img_correct += torch.sum(img_pred == target.data)
            mesh_correct += torch.sum(mh_pred == target.data)

            if (iteration%args.lr_step) == 0:
                lr = args.lr * (0.1 ** (iteration // args.lr_step))
                print('New  Learning Rate:     ' + str(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # update the learning rate of the center loss
            if (iteration%args.lr_step) == 0:
                lr_center = args.lr_center * (0.1 ** (iteration // args.lr_step))
                print('New  Center LR:     ' + str(lr_center))
                for param_group in optimizer_centloss.param_groups:
                    param_group['lr'] = lr_center

           
            if((iteration+1) % args.per_save) ==0:
                print('----------------- Save The Network ------------------------')
                with open(args.save + str(iteration+1)+'-img_global_net.pkl', 'wb') as f:
                    torch.save(img_net, f)
                with open(args.save + str(iteration+1)+'-pt_global_net.pkl', 'wb') as f:
                    torch.save(pt_net, f)
                with open(args.save + str(iteration+1)+'-mesh_global_net.pkl', 'wb') as f:
                    torch.save(mesh_net, f)

            iteration = iteration + 1
            batch_id = batch_id + 1

        epoch_point_acc = point_correct / len(train_set)
        epoch_img_acc = img_correct / len(train_set)
        epoch_mesh_acc = mesh_correct / len(train_set)
        
        log_string("epoch:%d loss:%.4f ce_loss:%.4f cmc_loss:%.4f mse_loss:%.4f"%(epoch,loss.item(),ce_loss.item(),cmc_loss.item(),mse_loss.item()))
        log_string("Point Cloud Train Accuracy:%.4f" % (epoch_point_acc))
        log_string("Image Train Accuracy: %.4f" % (epoch_img_acc))
        log_string("Mesh Train Accuracy: %.4f" % (epoch_mesh_acc))


        with torch.no_grad():
            test(img_net.eval(),mesh_net.eval(),pt_net.eval(),test_data_loader_loader,test_set)
    

        
 

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--dataset', type=str, default='ModelNet40', metavar='dataset',
                        help='ModelNet10 or ModelNet40')

    parser.add_argument('--dataset_dir', type=str, default='./dataset/', metavar='dataset_dir',
                        help='dataset_dir')

    parser.add_argument('--num_classes', type=int, default=40, metavar='num_classes',
                        help='10 or 40 or 8')

    parser.add_argument('--batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch')

    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of episode to train ')
    #optimizer
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--lr_step', type=int,  default=20000,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--lr_center', type=float, default=0.001, metavar='LR',
                        help='learning rate for center loss (default: 0.5)')
                                         
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    #DGCNN
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    # 注意 num_points 调整成为了2048
    #loss
    parser.add_argument('--weight_center', type=float, default=1.0, metavar='weight_center',
                        help='weight center (default: 1.0)')

    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--per_save', type=int,  default=1000,
                        help='how many iterations to save the model')

    parser.add_argument('--per_print', type=int,  default=100,
                        help='how many iterations to print the loss and accuracy')

    parser.add_argument('--k', type=int, default=20, help='it is used in pointcloud')
    parser.add_argument('--dropout', type=float, default=0.4, help='The argument in dropout')
    parser.add_argument('--emb_dims', type=int,default=512)

    parser.add_argument('--save', type=str,  default='./checkpoints/ModelNet40/GF/ce10_cmc1_mse01/',
                        help='path to save the final model')

    parser.add_argument('--gpu_id', type=str,  default='0', 
                        help='GPU used to train the network')

    parser.add_argument('--log', type=str,  default='ce10_cmc1_mse01',
                        help='path to the log information')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = False
    training(args)
    #test(args)
