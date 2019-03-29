import torch.nn as nn
import numpy as np
import torch

class DiscriminatorImg(nn.Module):
    def __init__(self, in_ch=3,  image_size=128, en_ch=50, num_exps=6,
                 conv_dim=64, repeat_num=4):
        super(DiscriminatorImg, self).__init__()
        self._name = 'DiscriminatorImage'
#        norm_fn = nn.BatchNorm2d
        norm_fn = nn.InstanceNorm2d
        
        layers = []
        layers.append(nn.Conv2d(3+num_exps, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(norm_fn(conv_dim, affine=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.conv0 = nn.Sequential(*layers)
        
        curr_dim = conv_dim
        layers = []
        layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
        layers.append(norm_fn(curr_dim*2, affine=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.conv1 = nn.Sequential(*layers)
        
        curr_dim = curr_dim * 2
        layers = []
        for i in range(0, repeat_num):
            out_dim = curr_dim * 2
            if i == 0:
                curr_dim = curr_dim + en_ch
                
            layers.append(nn.Conv2d(curr_dim, out_dim, kernel_size=4, stride=2, padding=1))
            layers.append(norm_fn(out_dim, affine=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = out_dim

#        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.d_out = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x, z, y):
        out = x
        out = concat_label(out, y)
        out = self.conv0(out)
#        print(out.size())
        out = self.conv1(out)
#        print(out.size())
#        print('z,',z.size())
        out = torch.cat([out, z],1)
#        print(out.size())
        out = self.main(out)
#        print(out.size())
        out = self.d_out(out)
#        print(out.size())
        return out.squeeze()
        
       
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
    def __init__(self, in_ch=3, en_ch=50, conv_dim=64, repeat_num=3):
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
        for i in range(2):
            layers = []
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(norm_fn(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
            setattr(self, 'up%d'%(i+1), nn.Sequential(*layers))
#        self.up = nn.Sequential(*layers)

        # Bottleneck
        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.middle = nn.Sequential(*layers)

        # Down-Sampling
#        layers = []
#        for i in range(3):
#            layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
#            layers.append(norm_fn(curr_dim, affine=True))
#            layers.append(nn.ReLU(inplace=True))
##            curr_dim = curr_dim // 2
#        self.down = nn.Sequential(*layers)
        
        # Out
        self.e_out = nn.Sequential(
                nn.Conv2d(curr_dim, en_ch, kernel_size=3, stride=1, padding=1, bias=False),
                norm_fn(en_ch, affine=True),
#                nn.Tanh(),
                nn.ReLU(inplace=True)
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
#        out = self.up3(out)
#        print(out.size())
#        feat3 = out # torch.Size([10, 512, 16, 16])
        out = self.middle(out)
#        print(out.size())
#        out = self.down(out)
#        print(out.size())
        feat3 = out
        out = self.e_out(out)
#        print(out.size())
#        out = out.view(out.size(0),-1)
#        out = self.fc(out)
#        out = out.view(out.size(0),-1)
#        print(out.size())
        if getFeat:
            return feat0, feat1, feat2, feat3
        else:
            return out


class Generator_z(nn.Module):
    def __init__(self, z_dim=50, conv_dim=64, en_ch=256):
        super(Generator_z, self).__init__()
        self.name = 'Generator_z'
        
#        norm_fn = nn.BatchNorm2d
        norm_fn = nn.InstanceNorm2d
        
        curr_dim = z_dim
        # Up-Sampling
        layers = []
        layers.append(nn.ConvTranspose2d(curr_dim, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(norm_fn(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(3):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(norm_fn(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        self.up = nn.Sequential(*layers)
        
        self.gz_out = nn.Sequential(
                nn.Conv2d(curr_dim, en_ch, kernel_size=3, stride=1, padding=1, bias=False),
                norm_fn(en_ch, affine=True),
                nn.ReLU(inplace=True)
                )
    
    def forward(self, z):
        out = z
#        print(out.size())
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out.expand(out.size(0),out.size(1), 2, 2)
        
        out = self.up(out)
#        print(out.size())
        out = self.gz_out(out)
#        print(out.size())
        return out

class Generator(nn.Module):
    def __init__(self, out_ch=3, en_ch=50, num_exps=6,
                 conv_dim=256, repeat_num=3):
        super(Generator, self).__init__()
        self._name = 'Generator'
        
#        norm_fn = nn.BatchNorm2d
        norm_fn = nn.InstanceNorm2d
        
        curr_dim = en_ch + num_exps
        
        
        self.conv0 = nn.Sequential(
                nn.Conv2d(curr_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False),
                norm_fn(conv_dim, affine=True),
                nn.ReLU()
                )
        curr_dim = conv_dim
        
        # Bottleneck
        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.middle = nn.Sequential(*layers)
        
        # Up-Sampling
        layers = []
        for i in range(2):
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
        out = concat_label(x, y)
#        out = out.view(out.size(0),out.size(1), 1, 1)
#        out = out.expand(out.size(0),out.size(1), 4, 4)
#        print(out.size())
#        out = self.up(out)
#        print(out.size())
        out = self.conv0(out)
#        print(out.size())
        out = self.middle(out)
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
#    print(label.size())
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
    en_ch = 128
    size_image = 128
    num_classes = 6
    num_z_ch = 50
    
    x = Variable(torch.rand([10,img_channel,size_image,size_image]))
    y = Variable(torch.rand([10,num_classes]))
    z = Variable(torch.rand([10,num_z_ch]))
    
    E = Encoder(in_ch=img_channel, en_ch=en_ch)
    G = Generator(out_ch=img_channel, en_ch=en_ch, num_exps=num_classes)
    Gz = Generator_z(z_dim=num_z_ch, en_ch=en_ch)
    Dimg = DiscriminatorImg(in_ch=img_channel, en_ch=en_ch, num_exps=num_classes)
    Dz = DiscriminatorZ(num_z_ch=num_z_ch, num_chs=[64, 32, 16], enable_bn=True)
    
    print("Encoder")
    out_e = E(x)
    print("Generator")
    out = G(out_e, y)
    print("Generator_z")
    out = Gz(z)
    print("DiscriminatorImg")
    out = Dimg(x, out_e, y)
#    print("DiscriminatorZ")
#    out = Dz(z)