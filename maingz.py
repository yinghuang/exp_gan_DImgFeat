#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
#import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import argparse
from modelgz import Generator, Generator_z, Encoder, DiscriminatorImg
import numpy as np
import socket
hostname = socket.gethostname()

seed = 2018

parser = argparse.ArgumentParser(description='Pytorch train_main')
parser.add_argument('--dataset', type=str, default='CK', choices=['CK', 'OULU', 'MMI', 'RAF', 'AffectNet'])
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=24, help='input batch size')
parser.add_argument('--train_G_every_n_iterations', type=int, default=5, help='train G every n interations')
parser.add_argument('--train_D_every_n_iterations', type=int, default=1, help='train D every n interations')
parser.add_argument('--display_interval', default=50, type=int, help='display train message interval')
parser.add_argument('--visual_interval', default=100, type=int, help='save sample imgs interval')      


parser.add_argument('--lambda_tv', type=float, default=0.0001, help='lambda smooth loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='lambda reconstruction loss')
parser.add_argument('--lambda_gan', type=float, default=1.0, help='lambda for real/fake discriminator loss')
parser.add_argument('--lambda_D_gp', type=float, default=10, help='lambda gradient penalty loss')
parser.add_argument('--encode_channel', type=int, default=256, help='dims of encoded features')
parser.add_argument('--z_dim', type=int, default=32, help='dims of encoded features')

parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for G adam')
parser.add_argument('--G_adam_b1', type=float, default=0.5, help='beta1 for G adam')
parser.add_argument('--G_adam_b2', type=float, default=0.999, help='beta2 for G adam')
parser.add_argument('--lr_D', type=float, default=0.0001, help='initial learning rate for D adam')
parser.add_argument('--D_adam_b1', type=float, default=0.5, help='beta1 for D adam')
parser.add_argument('--D_adam_b2', type=float, default=0.999, help='beta2 for D adam')

parser.add_argument('--optimize_mode', type=str, default='DCGAN', choices=['DCGAN', 'WGAN'])

parser.add_argument('--fold', default=1, type=int, help='cross validation fold')
parser.add_argument('--img_size', default=128, type=int, help='img size')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')

opt = parser.parse_args()
opt.is_train = True
opt.img_ch = 3








data_transform = transforms.Compose([
#            transforms.Grayscale(),
            transforms.Resize(opt.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.5],
#                                 std=[0.5])
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
    ])


data_root = '/media/data2/hy_data/data/exp_roi_datasets'
if hostname=="pami-ubuntu-server":
    data_root = '/media/data1/hy/exp_roi_datasets'# 59.77.15.88
    
from dataset import DatasetFolder
from dataset_AffectNet import Dataset as DatasetAffectNet
dsetname = opt.dataset
if dsetname=='RAF':
    data_dir = data_root+'/%s_100x100_nocrop'%(dsetname) 
#    data_dir = "/media/data2/hy_data/data/RAF/images"
    dset_train = DatasetFolder(os.path.join(data_dir, "train"),
                               land_des_dlib=None,
                               transform=data_transform,
                               pair=False)

    dset_test = DatasetFolder(os.path.join(data_dir, "test"),
                               land_des_dlib=None,
                               transform=data_transform,
                               pair=False)
elif dsetname=='AffectNet':
    
    TR_ANNO_FILE = "/media/data1/hy_data/data/affectnet/Manually_Annotated_file_lists/training.csv"
    VA_ANNO_FILE = "/media/data1/hy_data/data/affectnet/Manually_Annotated_file_lists/validation.csv"
    IMAGE_DIR = '/media/data1/hy_data/data/affectnet/cropped_Annotated'
    if hostname=="pami-ubuntu-server":# 59.77.15.88
        TR_ANNO_FILE = "/media/data1/hy/affectnet/Manually_Annotated_file_lists/training.csv"
        VA_ANNO_FILE = "/media/data1/hy/affectnet/Manually_Annotated_file_lists/validation.csv"
        IMAGE_DIR = '/media/data1/hy/affectnet/cropped_Annotated'
    dset_train = DatasetAffectNet(IMAGE_DIR,TR_ANNO_FILE, 
                            transform=data_transform, resize=240, pair=False)
    dset_test = DatasetAffectNet(IMAGE_DIR,VA_ANNO_FILE, 
                            transform=data_transform, resize=240, pair=False)

else:
    data_dir = data_root+'/%s_128x128_nocrop/cross_validation%s'%(dsetname,str(opt.fold))
    dset_train = DatasetFolder(os.path.join(data_dir, "train"),
                               land_des_dlib=None,
                               transform=data_transform,
                               pair=False)
    
    data_test_dir = data_root+'/%s_128x128_nocrop/cross_validation%s'%(dsetname,str(opt.fold))
    dset_test = DatasetFolder(os.path.join(data_test_dir, 'test_peak'),
                               land_des_dlib=None,
                               transform=data_transform,
                               pair=False)
    
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=opt.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=opt.batch_size, shuffle=True, num_workers=2)


opt.num_exps = len(dset_train.classes)
opt.checkpoints_dir = os.path.join(
        data_root,
        opt.checkpoints_dir + dsetname)
if dsetname in ['OULU','MMI','CK']:
    opt.checkpoints_dir = os.path.join(
            data_root,
            opt.checkpoints_dir,
            'fold_%d' % opt.fold)




print('------------ Options -------------')
for k, v in sorted(vars(opt).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# save options to txt
expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)
file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if opt.is_train else 'test'))
with open(file_name, 'wt') as opt_file:
    opt_file.write('------------ Options -------------\n')
    for k, v in sorted(vars(opt).items()):
        opt_file.write('%s: %s\n' % (str(k), str(v)))
    opt_file.write('-------------- End ----------------\n')




from collections import OrderedDict
import time
from PIL import Image, ImageDraw


class GAN():
    def __init__(self,opt):
        self._name = 'GAN'
        
        # create networks
        self.opt = opt
        self._save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self._init_nets()
        
        # init optimizers
        self._is_train = self.opt.is_train
        if self._is_train:
            self._init_optimizers()
        
        # load networks and optimizers
        if not self._is_train or self.opt.load_epoch > 0:
            self.load()
        
        # init loss values
        self._init_losses()
    
    def _init_nets(self):
        # encoder
        self._E = Encoder(in_ch=self.opt.img_ch, 
                          en_ch=self.opt.encode_channel)
        if len(self.opt.gpu_ids) > 1:
            self._E = torch.nn.DataParallel(self._E, device_ids=self.opt.gpu_ids)
        self._E.cuda()

        # generator
        self._G = Generator(out_ch=self.opt.img_ch, 
                            en_ch=self.opt.encode_channel,
                            num_exps=self.opt.num_exps)
        if len(self.opt.gpu_ids) > 1:
            self._G = torch.nn.DataParallel(self._G, device_ids=self.opt.gpu_ids)
        self._G.cuda()
        
        # generator for z
        self._Gz = Generator_z(z_dim=self.opt.z_dim,
                               en_ch=self.opt.encode_channel)
        if len(self.opt.gpu_ids) > 1:
            self._Gz = torch.nn.DataParallel(self._Gz, device_ids=self.opt.gpu_ids)
        self._Gz.cuda()
        
        # discriminator
        self._D = DiscriminatorImg(in_ch=self.opt.img_ch,  
                                   en_ch=self.opt.encode_channel, 
                                   num_exps=self.opt.num_exps)
        if len(self.opt.gpu_ids) > 1:
            self._D = torch.nn.DataParallel(self._D, device_ids=self.opt.gpu_ids)
        self._D.cuda()
    
    def _init_optimizers(self):
        self._optimize_mode = self.opt.optimize_mode
        self._current_lr_G = self.opt.lr_G
        self._current_lr_D = self.opt.lr_D

        # initialize optimizers
        EGparams = [
                {'params':self._Gz.parameters()},
                {'params':self._G.parameters()},
                {'params':self._E.parameters()},
            ]
        self._optimizer_EG = torch.optim.Adam(EGparams, lr=self._current_lr_G,
                                             betas=[self.opt.G_adam_b1, self.opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=[self.opt.D_adam_b1, self.opt.D_adam_b2])
    
    def set_input(self, input):
        self._input_real_img = input['real_img']
        self._input_real_y = input['real_y']
        self._input_z_prior = input['z_prior']
        self._input_fake_y = input['fake_y']
        
    def _init_losses(self):
        # define loss func
        self._criterion_cgan = torch.nn.BCEWithLogitsLoss().cuda()
        self._criterion_rec = torch.nn.L1Loss().cuda()
        def rec_loss(input, target):
            return torch.mean(torch.abs(input-target))
        def further_cont_loss(input, target):
            feat0_x, feat1_x, feat2_x, feat3_x = self._E(input, getFeat=True)
            feat0_y, feat1_y, feat2_y, feat3_y = self._E(target, getFeat=True)
#            l0 = torch.mean(torch.abs(feat0_x-(feat0_y.detach())))
#            l1 = torch.mean(torch.abs(feat1_x-(feat1_y.detach())))
#            l2 = torch.mean(torch.abs(feat2_x-(feat2_y.detach())))
            l0 = self._criterion_rec(feat0_x,feat0_y.detach())
            l1 = self._criterion_rec(feat1_x,feat1_y.detach())
            l2 = self._criterion_rec(feat2_x,feat2_y.detach())
#            l3 = torch.mean(torch.abs(feat3_x-feat3_y.detach()))
#            l4 = torch.mean(torch.abs(feat4_x-feat4_y.detach()))
            return (l0+l1+l2)/3.0
        self._criterion_rec_further = further_cont_loss
#        self._criterion_rec = rec_loss
        
        # init losses D
        self._loss_D_real = 0
        self._loss_D_fake = 0
        self._loss_D_gp = 0
        self._loss_D = 0
        
        self._loss_G_fake = 0
        self._loss_G_rec = 0
        self._loss_G_rec_fur = 0
        self._loss_G_tv = 0
        self._loss_G = 0
    
    def get_current_errors(self):
        loss_dict = OrderedDict([('g_fake', self._loss_G_fake),
                                 ('g_rec', self._loss_G_rec),
                                 ('g_rec_fur', self._loss_G_rec_fur),
                                 ('g_tv', self._loss_G_tv),
                                 ('g_sum', self._loss_G),
                                 
                                 ('d_real', self._loss_D_real),
                                 ('d_fake', self._loss_D_fake),
                                 ('d_gp', self._loss_D_gp),
                                 ('d_sum', self._loss_D)])

        return loss_dict
    
    def get_current_visuals(self):
        self._forward_EG(keep_data_for_visuals=True)
        vis_imgs_tensor = torch.cat([self._vis_real_imgs,
                                     self._vis_rec_imgs,
                                     self._vis_fake_imgs,
                                     self._vis_fake_imgs_from_prior])
        vis_names = ['real_imgs',
                     'rec_imgs',
                     'fake_imgs',
                     'fake_imgs_from_prior']
        vis_labels = torch.cat([self._vis_y,
                                self._vis_y,
                                self._vis_rand_y,
                                self._vis_rand_y])
        return vis_imgs_tensor, vis_names, vis_labels
        
    def set_train(self):
        self._E.train()
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._E.eval()
        self._G.eval()
        self._is_train = False
    
    def save_checkpoint(self, epoch):
#        if not os.path.isdir(self._save_dir):
#            os.mkdir(self._save_dir)
            
        state = {
            'Generator': self._G,
            'Discriminator': self._D,
            'Encoder': self._E,
            'epoch': epoch
        }
        save_path = os.path.join(self._save_dir,'%s_epoch_%d.t7'%(self.epoch))
        torch.save(state, save_path)
        print 'saved net: %s' % save_path
    
    def load(self, epoch):
        load_path = os.path.join(self._save_dir,'%s_epoch_%d.t7'%(self.epoch))
        checkpoint = torch.load(load_path)
        self._G = checkpoint['Generator']
        self._D = checkpoint['Discriminator']
        self._E = checkpoint['Encoder']
        print("resume from epoch:%d"%(epoch))
        
    def _compute_loss_gan(self, estim, is_real):
        if self._optimize_mode == "WGAN":
            return -torch.mean(estim) if is_real else torch.mean(estim)
        else:
            # cgan
            y_gan = torch.ones(estim.size()) if is_real else torch.zeros(estim.size())
            if self.opt.gpu_ids:
                y_gan = y_gan.cuda()
            return self._criterion_cgan(estim, Variable(y_gan))
        
    def _compute_loss_tv(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))
#        return (torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
#               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))) / (mat.size(2) * mat.size(3))
    
    def _gradinet_penalty_D(self, fake_imgs, gx, y):
        # interpolate sample
        alpha = torch.rand(fake_imgs.size(0), 1, 1, 1).cuda().expand_as(self._input_real_img)
        interpolated = Variable(alpha * self._input_real_img.data + (1 - alpha) * fake_imgs.data, requires_grad=True)
        interpolated_prob = self._D(interpolated, gx, y)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0].contiguous()

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        return torch.mean((grad_l2norm - 1) ** 2) 
    
    def _forward_D(self):
        # generate outputs
        gx_real = self._E.forward(self._input_real_img)
        imgs_fake = self._G.forward(gx_real, self._input_fake_y)
        d_logits_fake_img = self._D.forward(imgs_fake.detach(), gx_real.detach(), self._input_fake_y)
        d_logits_real = self._D.forward(self._input_real_img, gx_real.detach(), self._input_real_y)  
        
        gx_fake = self._Gz.forward(self._input_z_prior)
        d_logits_fake_gx = self._D.forward(self._input_real_img, gx_fake.detach(), self._input_real_y)  
        
        # loss gan
        self._loss_D_fake = self._compute_loss_gan(d_logits_fake_img, is_real=False)
        self._loss_D_fake += self._compute_loss_gan(d_logits_fake_gx, is_real=False)
        self._loss_D_fake *= 0.5
        self._loss_D_fake *= self.opt.lambda_gan
        
        self._loss_D_real = self._compute_loss_gan(d_logits_real, is_real=True) * self.opt.lambda_gan
        
        # Compute loss for gradient penalty
        self._loss_D_gp = self._gradinet_penalty_D(imgs_fake, 
                                                   gx_real, 
                                                   self._input_fake_y) * self.opt.lambda_D_gp
        
        # combine losses
        if self._optimize_mode == "WGAN":
            self._loss_D = self._loss_D_fake + self._loss_D_real + self._loss_D_gp
        else:
            self._loss_D = self._loss_D_fake + self._loss_D_real
        
        return self._loss_D
        
    def _forward_EG(self, stage=None, keep_data_for_visuals=False):
        # generate outputs
        gx_real = self._E.forward(self._input_real_img)
        imgs_fake_rec = self._G.forward(gx_real, self._input_real_y)
        imgs_fake = self._G.forward(gx_real, self._input_fake_y)
        d_logits_fake_img = self._D.forward(imgs_fake, gx_real, self._input_fake_y)
        
        gx_fake = self._Gz.forward(self._input_z_prior)
        d_logits_fake_gx = self._D.forward(self._input_real_img, gx_fake, self._input_real_y)  
        
        
        if keep_data_for_visuals:
            self._vis_real_imgs = self._input_real_img
            self._vis_rec_imgs = imgs_fake_rec
            self._vis_fake_imgs = imgs_fake
            self._vis_fake_imgs_from_prior = self._G.forward(gx_fake, self._input_fake_y)
            _, self._vis_rand_y = torch.max(self._input_fake_y, 1)
            _, self._vis_y = torch.max(self._input_real_y, 1)
        else:
            # loss gan
            self._loss_G_fake = self._compute_loss_gan(d_logits_fake_img, is_real=True)
            self._loss_G_fake += self._compute_loss_gan(d_logits_fake_gx, is_real=True)
            self._loss_G_fake *= 0.5
            self._loss_G_fake *= self.opt.lambda_gan
            
            # loss rec
            self._loss_G_rec = self._criterion_rec(imgs_fake, self._input_real_img) * self.opt.lambda_rec
            
            # loss rec further
            self._loss_G_rec_fur = self._criterion_rec_further(imgs_fake, self._input_real_img) * self.opt.lambda_rec * 0.0
            # loss tv
            self._loss_G_tv = self._compute_loss_tv(imgs_fake) * self.opt.lambda_tv
            
            # combine losses
            if stage == None or stage == 3:
                self._loss_G = self._loss_G_fake + self._loss_G_rec + self._loss_G_tv + self._loss_G_rec_fur
            elif stage == 1: # gan
                self._loss_G = self._loss_G_fake
            elif stage == 2: # rec  cls ?
                self._loss_G = self._loss_G_rec + self._loss_G_tv + self._loss_G_rec_fur
            return self._loss_G
    
    def optimize_params(self, stage=None, train_d=True, train_g=True):
        # train D
        if stage != 2:
            if train_d:
                loss_D = self._forward_D()
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()
        
        # train G
        if train_g:
            loss_G = self._forward_EG(stage=stage)
            self._optimizer_EG.zero_grad()
            loss_G.backward()
            self._optimizer_EG.step()
            
            
            
class Train:
    def __init__(self, dataset, opt):
        self.opt = opt
        self._gpu_ids = self.opt.gpu_ids
        self.train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
        self._iters_per_epoch = len(self.train_loader)
        self._model = GAN(opt)
        print("------------ E -------------")
        print(self._model._E)
        print("------------ G -------------")
        print(self._model._G)
        print("------------ Gz -------------")
        print(self._model._Gz)
        print("------------ D -------------")
        print(self._model._D)
        self._train()
        
    def tensor2var(self, x, volatile=True):
        """Convert torch tensor to variable."""
        if len(self._gpu_ids)>0:
            x = x.cuda()#(self._gpu_ids[0], async=True)
#        return Variable(x, volatile=volatile)
        return Variable(x)
    
    def label2onehot(self, y, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = y.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), y.long()] = 1
        return out
    
    def update_lr(self, optimizer, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_rand_data(self, batch_size, z_dim, num_classes):
        z_prior = torch.rand((batch_size, z_dim)) * 2 - 1 # -1 ~ 1
        y_fake = (torch.rand(batch_size, 1) * num_classes).type(torch.LongTensor)
        y_fake_hot = torch.zeros(batch_size, num_classes)
        y_fake_hot.scatter_(1, y_fake.view(batch_size, 1), 1)
        if len(self._gpu_ids)>0:
            z_prior = z_prior.cuda()#(self._gpu_ids[0], async=True)
            y_fake = y_fake.cuda()#(self._gpu_ids[0], async=True)
            y_fake_hot = y_fake_hot.cuda()#(self._gpu_ids[0], async=True)
        z_prior = Variable(z_prior)
        y_fake = y_fake.squeeze()
        y_fake = Variable(y_fake)
        y_fake_hot = Variable(y_fake_hot)
        return z_prior, y_fake, y_fake_hot
    
    
    #plt=show_result(1, input, y, 4, ['x_real',"x_rec","x_fake","x_fake_rand_y"],True)
    def visual_save(self, num_epoch, imgs_tensor, labels_tensor, rows, rows_name=None, show = False, img_path=None):
        n, c, h, w = imgs_tensor.size()
        imgs = imgs_tensor.cpu().data.numpy().transpose((0, 2, 3, 1))
        imgs = imgs.reshape(-1, h, w) if c ==1 else imgs.reshape(-1, h, w, 3)
        
        vmin = np.min(imgs)
        vmax = np.max(imgs)
        imgs = (imgs - vmin) * 255.0 / (vmax - vmin)
        
        cols = n / rows
        assert rows * cols == n 
        vis_img = np.zeros([rows * h, cols * w, c])
        
        for k in range(rows * cols):
            i = k // cols
            j = k % cols
            vis_img[h * i:h * (i + 1), w * j:w * (j + 1)] = imgs[k]
        vis_img = Image.fromarray(np.uint8(vis_img))
        
        # draw row name
        img_draw = ImageDraw.Draw(vis_img)
        if rows_name is not None:
            for idx, name in enumerate(rows_name):
                img_draw.text((0, h / 2 + h * idx), name)
        
        # draw bottom col name using labels
        labels = labels_tensor.cpu().data.numpy().reshape(-1)
        for idx, label in enumerate(labels):
            i = idx // cols
            j = idx % cols
            img_draw.text((w / 2 + w * j, h * i), str(label))
        
        # save
        if img_path:
            dirname = os.path.dirname(img_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            vis_img.save(img_path)
        return vis_img
#==============================================================================
#     def Get_confusion_matrix(true_label,pre_label):
#         acc = np.count_nonzero( np.equal(pre_label,true_label) ) * 1. / true_label.shape[0]
#         acc_mat = []
#         for i in np.unique(true_label):
#             index = np.where(true_label==i)
#             class_pre = pre_label[index]
#             class_num = len(index[0])
#             class_mat = []
#             for j in np.unique(true_label):
#                 pre_class_num = len(np.where(class_pre==j)[0]) *1.0
#                 class_mat.append ( pre_class_num/ class_num )
#             acc_mat.append(class_mat)
#         acc_mat=np.array(acc_mat)
#         return acc_mat,acc
#==============================================================================
    
    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "yes" if visuals_were_stored else ""
        message_head = '%s (T(if do visual?)%s, epoch: %d, it: %d/%d, t/smpl: %.3fs) ' % (log_time, visuals_info, epoch, i, iters_per_epoch, t)
        message_loss = ''
        for k, v in errors.items():
            message_loss += '%s:%.3f ' % (k, v)
#        print('\n')
        print(message_head)
        print(message_loss)
#        with open(self._log_path, "a") as log_file:
#            log_file.write('%s\n' % message)
    
    def _train(self):
        for i_epoch in range(self.opt.train_epochs):
            self._train_epoch(i_epoch)
        
    def _train_epoch(self, i_epoch):
        stage = None
        if i_epoch < 20:
            stage = 1
        else:
            if i_epoch < 30:
                stage = 2
            else:
                stage = 3
        self._model.set_train()
        for iter_idx, (x, y) in enumerate(self.train_loader):
            iter_start_time = time.time()
            
            # display, visual, save flags
            do_print_terminal = iter_idx % self.opt.display_interval==0
            do_visual = iter_idx % self.opt.visual_interval == 0
            do_save = False
            
            y = y.type(torch.LongTensor)
            y_hot = self.label2onehot(y, self.opt.num_exps)
            
            x = self.tensor2var(x)
            y = self.tensor2var(y)
            y_hot = self.tensor2var(y_hot)
            
            z_prior, _, y_fake_hot = self.get_rand_data(x.size(0), self.opt.z_dim, self.opt.num_exps)
            
            # train model
            input = dict()
            input['real_img'] = x
            input['real_y'] = y_hot
            input['z_prior'] = z_prior
            input['fake_y'] = y_fake_hot
            self._model.set_input(input)
            train_g = ((iter_idx+1) % self.opt.train_G_every_n_iterations == 0)
            train_d = ((iter_idx+1) % self.opt.train_D_every_n_iterations == 0)
            self._model.optimize_params(stage,train_d,train_g)
            
            # print terminal
            if do_print_terminal:
                errors = self._model.get_current_errors()
                t = (time.time() - iter_start_time) / self.opt.batch_size
                print('\nStage: %d'%stage)
                self.print_current_train_errors(i_epoch, iter_idx, self._iters_per_epoch, errors, t, do_visual)
                
            # display and save imgs
            if do_visual:
                imgs_tensor, imgs_name, labels_tensor = self._model.get_current_visuals()
                save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'sample_imgs', 
                                         'epoch_%d_iter_%d.png' % (i_epoch, iter_idx) )
                self.visual_save(i_epoch, imgs_tensor, labels_tensor, len(imgs_name), imgs_name, 
                                 False, save_path)
            # save model
            if do_save:
                a = 0


Train(dset_train, opt)








