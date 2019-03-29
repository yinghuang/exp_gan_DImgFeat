#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
import sys
import os

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def find_classes(dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class DatasetFolder(Dataset):
    def __init__(self, root, land_des_dlib=None, transform=None, target_transform=None, pair=False):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.land_des_dlib = land_des_dlib
        self.pair=pair

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        def get_item(index):
            path, target = self.samples[index]
# =============================================================================
#             while target == 0:
#                 index = np.random.randint(0, len(self.samples))
#                 path, target = self.samples[index]
# =============================================================================
            sample = self.loader(path)
            
            sample_gray = F.to_grayscale(sample)
            img_array = np.array(sample_gray,dtype=np.uint8)
            h,w, = np.shape(img_array)
        
#            rect = dlib.rectangle(0,0,img_array.shape[0],img_array.shape[1])
#            landmarks = np.matrix([[p.x, p.y] for p in self.land_des_dlib(img_array,rect).parts()])
#            ldmks = np.array(landmarks)
            
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        
        sample, target = get_item(index)
        
        if self.pair:
            # get same sample with same target
            index_p = np.random.randint(0, len(self.samples))
            sample_p, target_p = get_item(index_p)
            while (target_p!=target or index_p==index):
                index_p = np.random.randint(0, len(self.samples))
                sample_p, target_p = get_item(index_p)
            return sample, target, sample_p
        else:
            return sample, target
#        return sample, target, ldmks

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



#==============================================================================
# data_aug_train = transforms.Compose([
# #            transforms.Grayscale(),
#             transforms.Scale(128),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
# #            transforms.Normalize(mean=[0.5],
# #                                 std=[0.5])
#             transforms.Normalize(mean=[0.5,0.5,0.5],
#                                  std=[0.5,0.5,0.5])
#     ])
# data_aug_test = transforms.Compose([
# #        transforms.Grayscale(),
#         transforms.Scale(128),
#         transforms.ToTensor(),
# #        transforms.Normalize(mean=[0.5],
# #                             std=[0.5])
#         transforms.Normalize(mean=[0.5,0.5,0.5],
#                              std=[0.5,0.5,0.5])
#     ])
#         
# import os
# import torch
# 
# 
# 
# dsetf = DatasetFolder(os.path.join('/media/hy/4DAD7A3AD9A6C349/RAF_Face_Database/basic/RAF', 'train'),
#                      transform=data_aug_train)
# loader = torch.utils.data.DataLoader(dsetf, batch_size=10, shuffle=True, num_workers=2)
# 
# d_iter = iter(loader)
# x, y = next(d_iter)
#==============================================================================




#h5path = '/home/hy/e/Oulu-CASIA/ten_fold_ID_ascending/VL_Acropped/Strong/Oulu_VL_10fold_aligned_resize128/testfold1_train_no_ne_triplet.hdf5'
#==============================================================================
# h5path = '/home/hy/e/CK+/ten_fold_ID_ascending/128x128_nocrop/triplet_hdf5/testfold1_train_no_ne_with_co_triplet.hdf5'
# 
# dset = DatasetFromHdf5(h5path,transform=data_aug_train,land_des_dlib=predictor)
# loader = torch.utils.data.DataLoader(dset, batch_size=10, shuffle=True, num_workers=2)
# 
# 
# 
# 
# 
# 
# d_iter = iter(loader)
# x, y, ldmk, xie, yie = next(d_iter)
# # =============================================================================
# # import torch.nn.functional as F
# # bbbb = F.upsample_bilinear(input_iv,[256,256]).data * 0.5 + 0.5
# # images1 = torch.cat([bbbb,bbbb,bbbb],1)
# # images1 = images1[:,0] * 299/1000 + images1[:,1] * 587/1000 + images1[:,2] * 114/1000
# # images1 = images1.unsqueeze(1)
# # =============================================================================
# 
# x = x * 0.5 +0.5
# topil = ToPILImage()
# a = topil(x[0])
# b = topil(x[1])
# c = topil(x[2])
# d = topil(x[3])
# e = topil(x[4])
# 
# # =============================================================================
# # b1 = topil(images1[0])
# # =============================================================================
# 
# 
# plist = [19, 24, 27, 31, 35, 36, 45, 48, 51, 54, 57, ]
# 
# 
# import matplotlib.patches as patches
# 
# img = np.uint8(x[0][0]*255)
# landmarks = ldmk[0]
# # =============================================================================
# # rect = dlib.rectangle(0,0,img.shape[0],img.shape[1])
# # landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rect).parts()])
# # =============================================================================
# import matplotlib.pyplot as plt
# fig,ax = plt.subplots(1)
# ax.imshow(img,cmap='gray')
# 
# plt.xticks([])
# plt.yticks([])
# plt.savefig('/home/hy/exp_raw.pdf')
# fig,ax = plt.subplots(1)
# ax.imshow(img,cmap='gray')
# 
# c = ['orange', 'deepskyblue', 'chocolate', 'royalblue', '#ffff00', 
#      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
#          
# loc = [17,19,21,
# 22,24,26,
# 27,#28, 30 for nose
# 31,35,
# 36,39,
# 42,45,
# 48,51,54,57,
# ]
# loc = [17,
# 26,
# 27,#28, 30 for nose
# 31,35,
# 39,
# 42,
# 48,51,54,57,
# ]
# 
# r_h = 128
# r_w = 128
# size = 128
# factor_h = size*1.0 / r_h
# factor_w = size*1.0 / r_w
# pl_raw = 20
# for i in loc:
#     x = landmarks[i,0]*factor_h
#     y = landmarks[i,1]*factor_w
#     co = 'r'
#     if i in [17,39]:
#         co = c[0]
#     elif i in [42,26]:
#         co = c[1]
#     elif i in [35,54]:
#         co = c[5]
#     elif i in [31,48]:
#         co = c[4]
#     elif i in [51,57]:
#         co = c[3]
#     ax.plot(x,y,marker='o',color=co)
#     ax.text(x,y,str(i),color='r')
#     
# x_show = []
# # =============================================================================
# # for idx in plist:
# #     p1_r = np.int(np.round(pl_raw * factor_h))
# #     p1 = ( np.int(np.round(landmarks[idx,0]*factor_h)), np.int(np.round(landmarks[idx,1]*factor_w)))
# #     rect = patches.Rectangle((p1[0]-p1_r,p1[1]-p1_r),p1_r*2,p1_r*2, edgecolor='r',facecolor='none')
# #     ax.add_patch(rect)
# # #    plt.figure()
# # #    plt.imshow(img[p1[1]-p1_r:p1[1]+p1_r,p1[0]-p1_r:p1[0]+p1_r],cmap='gray')
# # plt.show()
# # =============================================================================
# 
# 
# p1 = ((landmarks[17,0]+landmarks[39,0])/2,(landmarks[17,1]+landmarks[39,1])/2)
# p2 = ((landmarks[42,0]+landmarks[26,0])/2,(landmarks[42,1]+landmarks[26,1])/2)
# p3 = ((landmarks[27,0]+landmarks[27,0])/2,(landmarks[27,1]+landmarks[27,1])/2)
# p4 = ((landmarks[51,0]+landmarks[57,0])/2,(landmarks[51,1]+landmarks[57,1])/2)
# p5 = ((landmarks[31,0]+landmarks[48,0])/2,(landmarks[31,1]+landmarks[48,1])/2)
# p6 = ((landmarks[35,0]+landmarks[54,0])/2,(landmarks[35,1]+landmarks[54,1])/2)
# plist = [p1,p2,p3,p4,p5,p6]
# for idx,xy in enumerate(plist):
#     r = np.int(np.round(pl_raw * factor_h))
#     x, y = xy[0], xy[1]
#     x, y = ( np.int(np.round(x*factor_h)), np.int(np.round(y*factor_w)))
#     rect = patches.Rectangle((x-r,y-r),r*2,r*2, edgecolor=c[idx],facecolor='none')
#     ax.add_patch(rect)
#     
# plt.xticks([])
# plt.yticks([])
# plt.savefig('/home/hy/exp_region0.pdf')
# plt.show()
# fig,ax = plt.subplots(1)
# ax.imshow(img,cmap='gray')
# for idx,xy in enumerate(plist):
#     r = np.int(np.round(pl_raw * factor_h))
#     x, y = xy[0], xy[1]
#     x, y = ( np.int(np.round(x*factor_h)), np.int(np.round(y*factor_w)))
#     rect = patches.Rectangle((x-r,y-r),r*2,r*2, edgecolor=c[idx],facecolor=c[idx])
#     ax.add_patch(rect)
#     
# plt.xticks([])
# plt.yticks([])
# plt.savefig('/home/hy/exp_region1.pdf')
# plt.show()
# 
# 
# 
# http_url ="https://api-cn.faceplusplus.com/facepp/v3/detect" 
# api_key = "7kHA0wt_eoNcVl93vFS2nhR8oBEuxGxe"
# api_secret = "dao9ber4cUSHIK5eJ3s3BLZZv0UqssUp"
# data = {"api_key":api_key, "api_secret":api_secret, "return_landmark":"2"}
# files = {"image_file":img}
# 
# # =============================================================================
# # import requests  
# # from json import JSONDecoder  
# # response = requests.post(http_url, data=data, files=files)  
# # req_con = response.content.decode('utf-8')  
# # req_dict = JSONDecoder().decode(req_con)  
# # print (len(req_dict[u'faces'])) 
# #  
# # =============================================================================
# 
# 
# filepath = "/home/hy/e/avg_face.jpg"
# a = open(filepath,"rb")
#==============================================================================

