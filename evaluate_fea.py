import numpy as np
from numpy.lib.function_base import append
import torch
from torch.autograd import Variable
import os
from tools.test_dataloader import TestDataloader
import argparse
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from util.append_feature import append_feature
import tqdm
device = torch.device("cuda")

def calc_map_label(source, target, label_test,name="name"): 
    source = normalize(source, norm='l1', axis=1)
    target = normalize(target, norm='l1', axis=1)
    dist = cdist(source, target, 'cosine') # cosine
    ord = dist.argsort()
    num = dist.shape[0]
    res = []
    for i in range(num):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(num):
            if label_test[i] == label_test[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]  
        else:
            res += [0]
    mAP = np.mean(res)
    return mAP

def extract_feature(args):

    test_set = TestDataloader(dataset=args.dataset, num_points = args.num_points , dataset_dir = args.dataset_dir, partition= 'test')
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=False, num_workers=8)
   
    if args.dataset == 'ModelNet40':

        # GF Extractor [512]
        img_global_net = torch.load("./checkpoints_1024/ModelNet40/GF_Extractor_PKL_40/best-img_global_net.pkl")
        img_global_net = img_global_net.eval()
        pt_global_net = torch.load("./checkpoints_1024/ModelNet40/GF_Extractor_PKL_40/best-pt_global_net.pkl")
        pt_global_net = pt_global_net.eval()
        mesh_global_net = torch.load("./checkpoints_1024/ModelNet40/GF_Extractor_PKL_40/best-mesh_global_net.pkl")
        mesh_global_net = mesh_global_net.eval() 

        #LFS Extractor
        img_local_net = torch.load("./checkpoints_1024/ModelNet40/LFS_Extractor_PKL_40/best-img_local_net.pkl")
        img_local_net = img_local_net.eval()
        pt_local_net = torch.load("./checkpoints_1024/ModelNet40/LFS_Extractor_PKL_40/best-pt_local_net.pkl")
        pt_local_net = pt_local_net.eval()
        mesh_local_net = torch.load("./checkpoints_1024/ModelNet40/LFS_Extractor_PKL_40/best-mesh_local_net.pkl")
        mesh_local_net = mesh_local_net.eval()
   
    if args.dataset == 'ModelNet10':
        
        # GF Extractor
        img_global_net = torch.load("./checkpoints/ModelNet10/GF_Extractor_PKL/45000_img_gf_net.pkl")
        img_global_net = img_global_net.eval()
        pt_global_net = torch.load("./checkpoints/ModelNet10/GF_Extractor_PKL/45000_pt_gf_net.pkl")
        pt_global_net = pt_global_net.eval()
        mesh_global_net = torch.load("./checkpoints/ModelNet10/GF_Extractor_PKL/45000_mesh_gf_net.pkl")
        mesh_global_net = mesh_global_net.eval()

        img_local_net = torch.load("./checkpoints/ModelNet10/LFS_Extractor_PKL/350-img_local_net.pkl")
        pt_local_net = torch.load("./checkpoints/ModelNet10/LFS_Extractor_PKL/350-pt_local_net.pkl")
        mesh_local_net = torch.load("./checkpoints/ModelNet10/LFS_Extractor_PKL/350-mesh_local_net.pkl")
        img_local_net = img_local_net.eval()
        pt_local_net = pt_local_net.eval()
        mesh_local_net = mesh_local_net.eval()


       
    img_feature_set = None
    img_1_feature_set = None
    img_2_feature_set = None
    mesh_feature_set = None
    point_feature_set = None

    img_global_feature_set = None
    img_1_global_feature_set = None
    img_2_global_feature_set = None
    mesh_global_feature_set = None
    point_global_feature_set = None

    img_local_feature_set = None
    img_1_local_feature_set = None
    img_2_local_feature_set = None
    mesh_local_feature_set = None
    point_local_feature_set = None


    label_feature_set = None

    batch_id = 0
    
    for data in test_data_loader:
        print("batch: %d/%d" % (batch_id,len(test_data_loader)))
        pt, img_list, centers, corners, normals, neighbor_index, target = data

        img_v1,img_v2,img_v3,img_v4 = img_list
        views = np.stack(img_list,axis=1)
        views = torch.from_numpy(views).to('cuda')

        img_list_1 = []
        img_list_1.append(img_v1)
        views_1 = np.stack(img_list_1,axis=1)
        views_1 = torch.from_numpy(views_1).to('cuda')

        img_list_2 = []
        img_list_2.append(img_v1)
        img_list_2.append(img_v2)
        views_2 = np.stack(img_list_2, axis=1)
        views_2 = torch.from_numpy(views_2).to('cuda')

        img_v1 = Variable(img_v1).to('cuda')
        img_v2 = Variable(img_v2).to('cuda')
        img_v3 = Variable(img_v3).to('cuda')
        img_v4 = Variable(img_v4).to('cuda')
        
        pt = Variable(pt).to('cuda')
        pt = pt.permute(0,2,1)

        target = target[:,0]
        target = Variable(target).to('cuda')

        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
        neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))

        torch.backends.cudnn.enabled = True
        img_global_feat_1 = img_global_net(img_v1, img_v1)
        img_global_feat_2 = img_global_net(img_v1, img_v2)
        img_global_feat = 0.5*(img_global_net(img_v1,img_v2)+img_global_net(img_v3,img_v4))
        mesh_global_feat = mesh_global_net(centers,corners,normals,neighbor_index)
        pt_global_feat = pt_global_net(pt)

        #torch.backends.cudnn.enabled = False
        img_local_feat_1 = img_local_net(views_1)
        img_local_feat_2 = img_local_net(views_2)
        img_local_feat = img_local_net(views)
        mesh_local_feat = mesh_local_net(centers,corners,normals,neighbor_index)
        pt_local_feat = pt_local_net(pt)

 
        label_feature_set = append_feature(label_feature_set,target.cpu().data.numpy(),flatten = True)

        img_1_global_feature_set = append_feature(img_1_global_feature_set,img_global_feat_1.cpu().data.numpy())
        img_2_global_feature_set = append_feature(img_2_global_feature_set,img_global_feat_2.cpu().data.numpy())
        img_global_feature_set = append_feature(img_global_feature_set,img_global_feat.cpu().data.numpy())
        mesh_global_feature_set = append_feature(mesh_global_feature_set, mesh_global_feat.cpu().data.numpy())
        point_global_feature_set = append_feature(point_global_feature_set,pt_global_feat.cpu().data.numpy())

        img_1_local_feature_set = append_feature(img_1_local_feature_set,img_local_feat_1.cpu().data.numpy())
        img_2_local_feature_set = append_feature(img_2_local_feature_set,img_local_feat_2.cpu().data.numpy())
        img_local_feature_set = append_feature(img_local_feature_set,img_local_feat.cpu().data.numpy())
        mesh_local_feature_set = append_feature(mesh_local_feature_set,mesh_local_feat.cpu().data.numpy())
        point_local_feature_set = append_feature(point_local_feature_set,pt_local_feat.cpu().data.numpy())
        
        
        
        img_feat = torch.cat((img_local_feat,img_global_feat),dim=1)
        img_feat_1 = torch.cat((img_local_feat_1,img_global_feat_1),dim=1)
        img_feat_2 = torch.cat((img_local_feat_2,img_global_feat_2),dim=1)
        pt_feat = torch.cat((pt_local_feat,pt_global_feat),dim=1)
        mesh_feat = torch.cat((mesh_local_feat,mesh_global_feat),dim=1)

        img_feature_set = append_feature(img_feature_set,img_feat.cpu().data.numpy())
        img_1_feature_set = append_feature(img_1_feature_set,img_feat_1.cpu().data.numpy())
        img_2_feature_set = append_feature(img_2_feature_set,img_feat_2.cpu().data.numpy())
        mesh_feature_set = append_feature(mesh_feature_set, mesh_feat.cpu().data.numpy())
        point_feature_set = append_feature(point_feature_set,pt_feat.cpu().data.numpy())
        batch_id = batch_id + 1

    # fusion feature
    np.save(args.save+'/ff_img_feat_{}'.format(1),img_1_feature_set)
    np.save(args.save+'/ff_img_feat_{}'.format(2),img_2_feature_set)
    np.save(args.save+'/ff_img_feat_{}'.format(4),img_feature_set)
    np.save(args.save+'/ff_pt_feat',point_feature_set)
    np.save(args.save+'/ff_mesh_feat',mesh_feature_set)
    
    # global feature
    np.save(args.save+'/gf_img_feat_{}'.format(1),img_1_global_feature_set)
    np.save(args.save+'/gf_img_feat_{}'.format(2),img_2_global_feature_set)
    np.save(args.save+'/gf_img_feat_{}'.format(4),img_global_feature_set)
    np.save(args.save+'/gf_pt_feat',point_global_feature_set)
    np.save(args.save+'/gf_mesh_feat',mesh_global_feature_set)


    # lfs
    np.save(args.save+'/lfs_img_feat_{}'.format(4),img_local_feature_set)
    np.save(args.save+'/lfs_pt_feat',point_local_feature_set)
    np.save(args.save+'/lfs_mesh_feat',mesh_local_feature_set)
    

    np.save(args.save+'/label',label_feature_set)

def eval_function(img_pairs):

    print("numver of img views: ",img_pairs)

    # gf_**_test:(2468,512) shape
    print("Global Feature") #[2468,512]
    gf_img_test = np.load(args.save+'/gf_img_feat_{}.npy'.format(img_pairs))
    gf_cloud_test = np.load(args.save+'/gf_pt_feat.npy')
    gf_mesh_test = np.load(args.save+'/gf_mesh_feat.npy')  
    gf_par_list = [
    (gf_img_test,gf_img_test,'Image2Image'), 
    (gf_img_test,gf_mesh_test,'Image2Mesh'),        
    (gf_img_test,gf_cloud_test,'Image2Point'), 
    (gf_mesh_test,gf_mesh_test,'Mesh2Mesh'),
    (gf_mesh_test,gf_img_test,'Mesh2Image'),
    (gf_mesh_test,gf_cloud_test,'Mesh2Point'),
    (gf_cloud_test,gf_cloud_test,'Point2Point'),
    (gf_cloud_test,gf_img_test,'Point2Image'),
    (gf_cloud_test,gf_mesh_test,'Point2Mesh')]

    # ff_**_test:(2468,1024) shape
    ff_img_test = np.load(args.save+'/ff_img_feat_{}.npy'.format(img_pairs))    
    ff_cloud_test = np.load(args.save+'/ff_pt_feat.npy')    
    ff_mesh_test = np.load(args.save+'/ff_mesh_feat.npy')  

    print("fusion feature:",ff_img_test.shape) #[2468,1024]
    ff_par_list = [
    (ff_img_test,ff_img_test,'Image2Image'), 
    (ff_img_test,ff_mesh_test,'Image2Mesh'),        
    (ff_img_test,ff_cloud_test,'Image2Point'), 
    (ff_mesh_test,ff_mesh_test,'Mesh2Mesh'),
    (ff_mesh_test,ff_img_test,'Mesh2Image'),
    (ff_mesh_test,ff_cloud_test,'Mesh2Point'),
    (ff_cloud_test,ff_cloud_test,'Point2Point'),
    (ff_cloud_test,ff_img_test,'Point2Image'),
    (ff_cloud_test,ff_mesh_test,'Point2Mesh')]
    print("Fusion Feature")

    # label shape(2468,1)
    label = np.load(args.save+'/label.npy') 
    print("label.shape", label.shape)

    for index in range(9):
        view_1, view_2,name = ff_par_list[index]
        acc = calc_map_label(view_1,view_2,label,name=name)
        acc_round = round(acc*100,2)
        print(name+' --- '+str(acc_round))
    
    print("Global Feature")
    for index in range(9):
        view_1, view_2,name = gf_par_list[index]
        acc = calc_map_label(view_1,view_2,label,name=name)
        acc_round = round(acc*100,2)
        print(name+' --- '+str(acc_round))
    

    print("Local Feature Set")
    # lfs_**_test shape (2468,512)
    
    lfs_img_test = np.load(args.save+'/lfs_img_feat_{}.npy'.format(img_pairs))
    lfs_cloud_test = np.load(args.save+'/lfs_pt_feat.npy')
    lfs_mesh_test = np.load(args.save+'/lfs_mesh_feat.npy')  
    lfs_par_list = [
        (lfs_img_test,lfs_img_test,'Image2Image'), 
        (lfs_img_test,lfs_mesh_test,'Image2Mesh'),        
        (lfs_img_test,lfs_cloud_test,'Image2Point'), 
        (lfs_mesh_test,lfs_mesh_test,'Mesh2Mesh'),
        (lfs_mesh_test,lfs_img_test,'Mesh2Image'),
        (lfs_mesh_test,lfs_cloud_test,'Mesh2Point'),
        (lfs_cloud_test,lfs_cloud_test,'Point2Point'),
        (lfs_cloud_test,lfs_img_test,'Point2Image'),
        (lfs_cloud_test,lfs_mesh_test,'Point2Mesh')]
    for index in range(9):
        view_1, view_2,name = lfs_par_list[index]
        acc = calc_map_label(view_1,view_2,label)
        acc_round = round(acc*100,2)
        print(name+' --- '+str(acc_round)) 
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--dataset', type=str, default='ModelNet40', metavar='dataset',
                        help='ModelNet10 or ModelNet40')

    parser.add_argument('--per_test', type=int, default=20)

    parser.add_argument('--dataset_dir', type=str, default='./dataset/', metavar='dataset_dir',
                        help='dataset_dir')

    parser.add_argument('--num_classes', type=int, default=40, metavar='num_classes',
                        help='10 or 40')

    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch')

    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of episode to train ')
    #optimizer
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--lr_step', type=int,  default=8000,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--lr_center', type=float, default=0.001, metavar='LR',
                        help='learning rate for center loss (default: 0.5)')
                                         
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    #DGCNN
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    
    #loss
    parser.add_argument('--weight_center', type=float, default=1.0, metavar='weight_center',
                        help='weight center (default: 1.0)')
    parser.add_argument('--weight_local_center', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    # parser.add_argument('--per_save', type=int,  default=5000,
    #                     help='how many iterations to save the model')
    parser.add_argument('--save', type=str,  default='extracted_features/ModelNet40_gf_512_lfs_512',
                        help='path to save the final model')

    parser.add_argument('--per_print', type=int,  default=100,
                        help='how many iterations to print the loss and accuracy')

    parser.add_argument('--k', type=int, default=20, help='it is used in pointcloud')

    parser.add_argument('--dropout', type=float, default=0.4, help='The argument in dropout')
    
    parser.add_argument('--emb_dims', type=int,default=512)

    parser.add_argument('--gpu_id', type=str,  default='4,5,6,7', 
                        help='GPU used to train the network')

    parser.add_argument('--log', type=str,  default='log/',
                        help='path to the log information')

    args = parser.parse_args()   
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

    test_set = TestDataloader(dataset=args.dataset, num_points = args.num_points, dataset_dir = args.dataset_dir, partition= 'test')
    test_data_loader_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=False, num_workers=8)
    
    #extract_feature(args)
    print("Eatract the Features Sucessfully")

    
    
    # eval_function(1)
    # eval_function(2)
    eval_function(4)
    