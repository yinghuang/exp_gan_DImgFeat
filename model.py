import torch.nn as nn
import numpy as np
import torch

class DiscriminatorImg(nn.Module):
    def __init__(self, in_ch=3,  image_size=128, num_z_ch=50, num_exps=6,
                 conv_dim=64, repeat_num=6):
        super(DiscriminatorImg, self).__init__()
        self._name = 'DiscriminatorImage'
#        norm_fn = nn.BatchNorm2d
        norm_fn = nn.InstanceNorm2d
        layers = []
        layers.append(nn.Conv2d(3+num_z_ch, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(norm_fn(conv_dim, affine=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(norm_fn(curr_dim*2, affine=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.d_out = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.d_out_cls = nn.Conv2d(curr_dim, num_exps, kernel_size=k_size, bias=False)
        
    def forward(self, x, z):
        out = x
#        out = concat_label(out, y)
        out = concat_label(out, z)
        h = self.main(out)
#        print(out.size())
        out = self.d_out(h)
#        print(out.size())
        out_cls = self.d_out_cls(h)
#        print(out_cls.size())
        return out.squeeze(), out_cls.squeeze()
        
       
class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
#        norm_fn = nn.BatchNorm2d
        norm_fn = nn.InstanceNorm2d
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)
    
    
class Encoder(nn.Module):
    def __init__(self, in_ch=3, num_z_ch=50, conv_dim=64, repeat_num=2):
        super(Encoder, self).__init__()
        self._name = 'Encoder'
        
        norm_fn = nn.BatchNorm2d
#        norm_fn = nn.InstanceNorm2d
        layers = []
        layers.append(nn.Conv2d(in_ch, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(norm_fn(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        self.up0 = nn.Sequential(*layers)
        # Down-Sampling
        for i in range(3):
            layers = []
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(norm_fn(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
            setattr(self, 'up%d'%(i+1), nn.Sequential(*layers))
#        self.up = nn.Sequential(*layers)

#==============================================================================
#         # Bottleneck
#         layers = []
#         for i in range(repeat_num):
#             layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
#         self.middle = nn.Sequential(*layers)
#==============================================================================

        # Down-Sampling
        layers = []
        for i in range(3):
            layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(norm_fn(curr_dim, affine=True))
            layers.append(nn.ReLU(inplace=True))
#            curr_dim = curr_dim // 2
        self.down = nn.Sequential(*layers)
        
        # Out
        self.e_out = nn.Sequential(
                nn.Conv2d(curr_dim, num_z_ch, kernel_size=4, stride=2, padding=1, bias=False),
                norm_fn(num_z_ch, affine=True),
                nn.Tanh()
                )
#        self.e_out = nn.Sequential(
#                nn.Linear(curr_dim*4*4, num_z_ch, bias=False),
#                nn.Tanh()
#                )
    def forward(self, x, getFeat=False):
        out = x
        out = self.up0(out)
#        print(out.size())
        feat0 = out # torch.Size([10, 64, 128, 128])
        out = self.up1(out)
#        print(out.size())
        feat1 = out # torch.Size([10, 128, 64, 64])
        out = self.up2(out)
#        print(out.size())
        feat2 = out # torch.Size([10, 256, 32, 32])
        out = self.up3(out)
#        print(out.size())
        feat3 = out # torch.Size([10, 512, 16, 16])
#        out = self.middle(out)
#        print(out.size())
        out = self.down(out)
#        print(out.size())
        feat4 = out
        out = self.e_out(out)
#        print(out.size())
#        out = out.view(out.size(0),-1)
#        out = self.fc(out)
        out = out.view(out.size(0),-1)
#        print(out.size())
        if getFeat:
            return feat0, feat1, feat2, feat3
        else:
            return out




class Generator(nn.Module):
    def __init__(self, out_ch=3, num_z_ch=50, num_exps=6,
                 conv_dim=512, repeat_num=2,
                 tile_ratio=1.0, enable_tile_label=False):
        super(Generator, self).__init__()
        self._name = 'Generator'
        
        norm_fn = nn.BatchNorm2d
#        norm_fn = nn.InstanceNorm2d
        
        if enable_tile_label:
            self.duplicate = int(num_z_ch * tile_ratio / num_exps)
            curr_dim = num_z_ch + num_exps * self.duplicate
        else:
            curr_dim = num_z_ch + num_exps
            self.duplicate = 1
        
        
        # Up-Sampling
        layers = []
        layers.append(nn.ConvTranspose2d(curr_dim, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(norm_fn(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(3):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim*1, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(norm_fn(curr_dim*1, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 1
        self.up = nn.Sequential(*layers)
        
#==============================================================================
#         # Bottleneck
#         layers = []
#         for i in range(repeat_num):
#             layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
#         self.middle = nn.Sequential(*layers)
#==============================================================================
        
        # Up-Sampling
        layers = []
        for i in range(3):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(norm_fn(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        self.down = nn.Sequential(*layers)
        
        # Out
        self.g_out = nn.Sequential(
                nn.Conv2d(curr_dim, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh()
                )
    def forward(self, x, y):
        out = x
        out = concat_label(x, y, duplicate=self.duplicate)
        out = out.view(out.size(0),out.size(1), 1, 1)
#        out = out.expand(out.size(0),out.size(1), 4, 4)
#        print(out.size())
        out = self.up(out)
#        print(out.size())
#        out = self.middle(out)
#        print(out.size())
        out = self.down(out)
#        print(out.size())
        out = self.g_out(out)
#        print(out.size())
        return out

class DiscriminatorZ(nn.Module):
    def __init__(self, num_z_ch=50, num_chs=[64, 32, 16], enable_bn=True):
        super(DiscriminatorZ, self).__init__()
        self._name = 'DiscriminatorZ'
        self.fc_name_pre = "dz_fc"
        
        
        act_fn = nn.ReLU()
        cur_dim = num_z_ch
        self.num_layers = len(num_chs)
        for i in range(self.num_layers):
            seq = []
            out_dim = num_chs[i]
            seq += [nn.Linear(cur_dim, out_dim)]
            if enable_bn:
                seq += [nn.BatchNorm1d(out_dim)]
            seq += [act_fn]
            cur_dim = out_dim
            setattr(self, self.fc_name_pre+"_%d"%(i+1), nn.Sequential(*seq))
        
        # 1
        seq = []
        out_dim = 1
        seq += [nn.Linear(cur_dim, out_dim)]
#                seq += [nn.Sigmoid]
        setattr(self, self.fc_name_pre+"_%d"%(i+2), nn.Sequential(*seq))
            
            
    def forward(self, z):
        out = z
        for i in range(self.num_layers):
            layer = getattr(self, self.fc_name_pre+"_%d"%(i+1))
            out = layer(out)
#            print(out.size())
            
        layer = getattr(self, self.fc_name_pre+"_%d"%(i+2))
        out = layer(out)
#        print(out.size())
        return out


def concat_label(x, label, duplicate=1):
    if duplicate < 1:
        return x
    label = label.repeat(1, duplicate)
    if len(x.size()) == 2:
        return torch.cat([x, label], 1)
    elif len(x.size()) == 4:
        label = label.unsqueeze(2).unsqueeze(3)
        label = label.expand(label.size(0), label.size(1), x.size(2), x.size(3))
        return torch.cat([x, label], 1)


def easy_deconv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)

    c_in, h_in, w_in = in_dims
    c_out, h_out, w_out = out_dims

    padding = [0, 0]
    output_padding = [0, 0]

    lhs_0 = -h_out + (h_in - 1) * stride[0] + kernel[0]  # = 2p[0] - o[0]
    if lhs_0 % 2 == 0:
        padding[0] = lhs_0 // 2
    else:
        padding[0] = lhs_0 // 2 + 1
        output_padding[0] = 1

    lhs_1 = -w_out + (w_in - 1) * stride[1] + kernel[1]  # = 2p[1] - o[1]
    if lhs_1 % 2 == 0:
        padding[1] = lhs_1 // 2
    else:
        padding[1] = lhs_1 // 2 + 1
        output_padding[1] = 1

    return torch.nn.ConvTranspose2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel,
        stride=stride,
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        groups=groups,
        bias=bias,
        dilation=dilation
    )
        
        
if __name__ == '__main__':
    from torch.autograd import Variable
    print("\n\nfor model_from_starGAN.py") 
    img_channel = 3
    f_dim = 64
    size_image = 128
    num_classes = 6
    num_en_ch = 64
    num_z_ch = 50
    
    x = Variable(torch.rand([10,img_channel,size_image,size_image]))
    y = Variable(torch.rand([10,num_classes]))
    z = Variable(torch.rand([10,num_z_ch]))
    
    E = Encoder(in_ch=img_channel, num_z_ch=num_z_ch)
    G = Generator(out_ch=img_channel, num_z_ch=num_z_ch, num_exps=num_classes)
    Dimg = DiscriminatorImg(in_ch=img_channel, num_z_ch=num_z_ch, num_exps=num_classes)
    Dz = DiscriminatorZ(num_z_ch=num_z_ch, num_chs=[64, 32, 16], enable_bn=True)
    
    print("Encoder")
    out = E(x)
    print("Generator")
    out = G(out, y)
    print("DiscriminatorImg")
    out, out_cls = Dimg(x, z)
    print("DiscriminatorZ")
    out = Dz(z)