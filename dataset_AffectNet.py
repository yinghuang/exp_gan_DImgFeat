# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import scipy.misc as sm
import torch
#from torchvision.transforms import functional as F
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import copy
from PIL import Image


def make_pair(images, labels):
    labels = np.array(labels)
    idxs_dict = {} 
    for _,y in enumerate(np.unique(labels)):
        idxs_dict[y]=list(np.where(labels==y)[0])
    
    images_pair = []
    labels_pair = []
    print('make_pair for the dataset...')
    for i in range(len(labels)):
        print(i,len(labels))
        pro = i*100/len(labels)
        if (pro is not 0) and (pro % 5 == 0):
            print("%d%% has completed."%(pro))
#        print("make_pair, %d/%d"%(i, len(labels)))
        y = labels[i]
        _idxs_raw = idxs_dict[y]
#        _idxs = _idxs_raw.copy() # for python3
        _idxs = copy.deepcopy(_idxs_raw) # for python2
        _idxs.remove(i) 
        if len(_idxs) == 0:
            _idxs = [i] 
            print('class %d has only one element' % y)
        _idx_for_idxs = np.random.randint(0, len(_idxs))
        pair_idx = _idxs[_idx_for_idxs] 
        if i == pair_idx:
            print('length: %d, now idx(from 0): %d, pair idx: %d, same!!!'%(len(labels), i, pair_idx))
        labels_pair.append(labels[pair_idx])
        images_pair.append(images[pair_idx])
    
    if not (labels==labels_pair).all():
        raise ValueError('labels_pair not equal to labels')
    return images_pair
    
class Dataset:
    def __init__(self, root, ano_path, resize=128, transform=None, target_transform=None, pair=False, vggface_trans=False):

        self.root_dir = root
        self.size = resize
        self.ano_df = pd.read_csv(ano_path)
        self.images = self.ano_df.loc[:]['subDirectory_filePath']
        self.labels = self.ano_df.loc[:]['expression']
        labels_arr = np.array(self.labels)
        self.num_cls = len(np.unique(labels_arr))
        self.vggface_trans = vggface_trans
#        self.images_pair = make_pair(list(self.images), list(self.labels))
#        self.loader =  sm.imread
        self.loader = default_loader
        self.num = len(self.ano_df)
        self.pair = pair
        self.transform = transform
        self.mean_rgb = np.array([131.0912, 103.8827, 91.4953],dtype=np.float32)  # from resnet50_ft.prototxt, RGB
        self.target_transform = target_transform
        self.filter()
        
    def filter(self):
# =============================================================================
# 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
# 7: Contempt, 8: None, 9: Uncertain, 10: No-Face
# =============================================================================
        select_labels = [0,1,2,3,4,5,6]
        self.classes = select_labels
        # select by condition indexing
        idx = (self.labels>float('Inf')) # all elemets are False
        for item in select_labels:
            idx |= (self.labels==item)
        self.images = self.images[idx]
        self.labels = self.labels[idx]
        self.ano_df = self.ano_df[idx]
# =============================================================================
#         # select by checking each element
#         for idx in range(len(self.labels)):
#             target = self.labels[idx]
#             if target not in select_labels:
#                 self.labels = self.labels.drop([idx])
#                 self.images = self.images.drop([idx])
# =============================================================================
        # reset index
        self.images = self.images.reset_index(drop=True)
        self.labels = self.labels.reset_index(drop=True)
        self.ano_df = self.ano_df.reset_index(drop=True)
        self.num_cls = len(np.unique(select_labels))
        self.num = len(self.images)
        
    def __getitem__(self, index):
        def get_item(index):
            img_path = self.ano_df.loc[index]['subDirectory_filePath']
            target = self.ano_df.loc[index]['expression']
            face_x = np.int32(self.ano_df.loc[index]['face_x'])
            face_y = np.int32(self.ano_df.loc[index]['face_y'])
            face_width = self.ano_df.loc[index]['face_width']
            face_height = self.ano_df.loc[index]['face_height']
            sample = self.loader(os.path.join(self.root_dir,img_path))
            if isinstance(sample, np.ndarray): # for m.imread loader
                cropped = sample[face_y:face_y+face_height, face_x:face_x+face_width]
                img = sm.imresize(cropped,size=[self.size,self.size])
            elif isinstance(sample, Image.Image): # for default_loader
                box = (face_x, face_y, face_x+face_width, face_y+face_height) #left, upper, right, lower
                cropped = sample.crop(box)
                size_re = [self.size, self.size] # width, heigth
                img = cropped.resize(size_re)
            
            if self.transform is not None:# transform should done on PIL.Image
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            # for VGG-FACE2 transform
            if self.vggface_trans:# don't do to tensor in transform
                img_out = np.array(img, dtype=np.float32)
                img_out -= self.mean_rgb
                img_out = img_out.transpose(2, 0, 1)  # C x H x W
                img_out = torch.from_numpy(img_out).float()
            else:
                img_out = img
                
            return img_out, target
        
        error = True
        while error:
            try:
                sample, target = get_item(index) # there are some file missed or broken
                error = False
            except Exception as e:
            # 29a31ebf1567693f4644c8ba3476ca9a72ee07fe67a5860d98707a0a.jpg # can't read for local and server
            # 9db2af5a1da8bd77355e8c6a655da519a899ecc42641bf254107bfc0.jpg # losed for local and server
                index += 1
#                print(e)
        
        if self.pair:
            # get same sample with same target
            index_p = np.random.randint(0, self.num)
            sample_p, target_p = get_item(index_p)
            while (target_p!=target or index_p==index):
                index_p = np.random.randint(0, self.num)
                sample_p, target_p = get_item(index_p)
#            sample_p = self.images_pair[index]
            return sample, target, sample_p
        else:
            return sample, target
    
    def __len__(self):
        return len(self.ano_df)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def to_pil(img_tensor):
    # for VGG-FACE2
    npimg = np.transpose(img_tensor.numpy(), (1, 2, 0))
    mean_rgb = np.array([131.0912, 103.8827, 91.4953],dtype=np.float32)  # from resnet50_ft.prototxt, RGB
    npimg += mean_rgb
    npimg = npimg.astype(np.uint8)
    mode = 'RGB'
    return Image.fromarray(npimg, mode=mode)


if __name__=='__main__':
    # 31e1ab1e8c47cbeba101848c8c4085a079cd5a4223f413f6231894f7.jpg
# =============================================================================
#     a = "../Manually_Annotated_compressed/cropped_Annotated/103/31e1ab1e8c47cbeba101848c8c4085a079cd5a4223f413f6231894f7.jpg"
#     error = True
#     while error:
#         try:
#             img = default_loader(a)
#             error = False
#         except Exception as e:
#             print(e)
# =============================================================================


    TR_ANNO_FILE = "../Manually_Annotated_file_lists/training.csv"
    VA_ANNO_FILE = "../Manually_Annotated_file_lists/validation.csv"
    IMAGE_DIR = '../Manually_Annotated_compressed/cropped_Annotated'
    vggface_trans = True
#=================vggface_trans
    if vggface_trans:
        torch_trans_tran = transforms.Compose([
                    transforms.Pad(padding=5),
                    transforms.RandomCrop(224, padding=0),
                    transforms.RandomHorizontalFlip(),
             ])
                 
        torch_trans_test = transforms.Compose([
                         transforms.CenterCrop(224),
             ])
    else:#use ToTensor
        torch_trans_tran = transforms.Compose([
                    transforms.Pad(padding=5),
                    transforms.RandomCrop(224, padding=0),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
             ])
                 
        torch_trans_test = transforms.Compose([
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
             ])
    
#    TR_ANNO_FILE = "/media/data1/hy_data/data/affectnet/Manually_Annotated_file_lists/training.csv"
#    IMAGE_DIR = '/media/data1/hy_data/data/affectnet/cropped_Annotated'
    dset = Dataset(IMAGE_DIR,TR_ANNO_FILE, 
                   transform=torch_trans_tran, resize=240, pair=False, vggface_trans=vggface_trans)

    print('num for: {} is: {}'.format(TR_ANNO_FILE, dset.num))
    loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=4)
    d_iter = iter(loader)
    x, y = next(d_iter)
    if vggface_trans:
        pilimg = to_pil(x[2])
    else:
        to_pil_image=transforms.ToPILImage()
        pilimg = to_pil_image(x[2])
        
        
        
#==============================================================================
#     iter_per_epoch = len(loader)
#     for step in range(10000):
#         # Reset the data_iter
#         if (step+1) % iter_per_epoch == 0:
#             d_iter = iter(loader)
#         x, y = next(d_iter)
#         print("epoch:{:d},step:{},size:{}".format(int(step/iter_per_epoch),step, x.size()))
#==============================================================================
#    for batch_idx, (x, y) in enumerate(loader):
#        print(batch_idx, x.size())
