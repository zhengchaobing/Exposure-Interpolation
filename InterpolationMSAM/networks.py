import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, bias=True, padding=1, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


#####Gradient Branch#####
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x

## channel attention modules
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## spatial  attention
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale
        ##########################################################################

###BasicConv
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

###ChannelPool
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)



##########################################################################
class gaussian_attn_layer(nn.Module):
    def __init__(self):
        super(gaussian_attn_layer, self).__init__()
        n_feats = 64
        self.Conv31 = conv(n_feats, n_feats, kernel_size=3, padding=(3//2),stride=1)
        self.Conv51 = conv(n_feats, n_feats, kernel_size=5, padding=(5//2),stride=1)
        self.Conv71 = conv(n_feats, n_feats, kernel_size=7, padding=(7//2),stride=1)

        self.prelu = nn.PReLU(64)
        self.sa = spatial_attn_layer()
        self.confusion = conv(n_feats*3, n_feats, kernel_size=1, padding=0,stride=1)


    def forward(self, x):
        A31 = self.prelu(self.Conv31(x))
        A51 = self.prelu(self.Conv51(x))
        A71 = self.prelu(self.Conv71(x))
        out1 = torch.cat((A31,A51,A71), dim=1)
        attention = self.sa(out1)*out1
        feature_fusion = self.confusion(attention) + x
        return feature_fusion


class ResBlock1(nn.Module):
    def __init__(self, conv, n_feat1=3, n_feat=64, kernel_size=3, bias=True, act=nn.PReLU(64)):
        super(ResBlock1, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat1, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class ResBlock2(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, act=nn.PReLU(64)):
        super(ResBlock2, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)+x
        return res

#######################################################################################################################################
#######################################################################################################################################
class Enhance(nn.Module):
    def __init__(self, channels=3, num_of_layers=8):
        super(Enhance, self).__init__()
        kernel_size = 3
        padding = 1
        features = 3
        layers1 = [];   layers2  = [];   layers3  = [];  layers4 = [];  layers5 = [];    layers6  = [];   layers7  = [];  layers8 = [];  layers9 = []; layers10 = [];
        layers11 = [];  layers12 = [];   layers13 = [];  layers14 = []; layers15 = [];   layers16 = [];   layers17 = [];  layers18 = []; layers19 = []; layers20 = [];
        layers1A = [];  layers2A = [];   layers3A = [];  layers4A = []; layers5A = []; layers6A = []; layers7A = []; layers8A = []; layers9A = []; layers10A = [];
        layers1B = [];  layers2B = [];   layers3B = [];  layers4B = []; layers5B = []; layers6B = []; layers7B = []; layers8B = []; layers9B = []; layers10B = [];


        layersEnd = []; layersEnd2 = [];
        n_feats = 64
        act = nn.PReLU(n_feats)

        layers1.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers1.append(act)
        layers1.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn1 = nn.Sequential(*layers1)

        layers2.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers2.append(act)
        layers2.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn2 = nn.Sequential(*layers2)

        layers3.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers3.append(act)
        layers3.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn3 = nn.Sequential(*layers3)

        layers4.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers4.append(act)
        layers4.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn4 = nn.Sequential(*layers4)

        layers5.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers5.append(act)
        layers5.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn5 = nn.Sequential(*layers5)

        layers6.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers6.append(act)
        layers6.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn6 = nn.Sequential(*layers6)

        layers7.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers7.append(act)
        layers7.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn7 = nn.Sequential(*layers7)

        layers8.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers8.append(act)
        layers8.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn8 = nn.Sequential(*layers8)

        layers9.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers9.append(act)
        layers9.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn9 = nn.Sequential(*layers9)

        layers10.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers10.append(act)
        layers10.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn10 = nn.Sequential(*layers10)
        ####################################################################################################################3
        layers11.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers11.append(act)
        layers11.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn11 = nn.Sequential(*layers11)

        layers12.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers12.append(act)
        layers12.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn12 = nn.Sequential(*layers12)

        layers13.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers13.append(act)
        layers13.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn13 = nn.Sequential(*layers13)

        layers14.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers14.append(act)
        layers14.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn14 = nn.Sequential(*layers14)

        layers15.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers15.append(act)
        layers15.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn15 = nn.Sequential(*layers15)

        layers16.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers16.append(act)
        layers16.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn16 = nn.Sequential(*layers16)

        layers17.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers17.append(act)
        layers17.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn17 = nn.Sequential(*layers17)

        layers18.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers18.append(act)
        layers18.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn18 = nn.Sequential(*layers18)

        layers19.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers19.append(act)
        layers19.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn19 = nn.Sequential(*layers19)

        layers20.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        layers20.append(act)
        layers20.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True));
        self.dncnn20 = nn.Sequential(*layers20)


        #######################################################################################################################
        #######################################################################################################################
        layers1A.append(gaussian_attn_layer());
        self.dncnn1A = nn.Sequential(*layers1A)

        layers2A.append(gaussian_attn_layer());
        self.dncnn2A = nn.Sequential(*layers2A)

        layers3A.append(gaussian_attn_layer());
        self.dncnn3A = nn.Sequential(*layers3A)

        layers4A.append(gaussian_attn_layer());
        self.dncnn4A = nn.Sequential(*layers4A)

        layers5A.append(gaussian_attn_layer());
        self.dncnn5A = nn.Sequential(*layers5A)

        layers6A.append(gaussian_attn_layer());
        self.dncnn6A = nn.Sequential(*layers6A)

        layers7A.append(gaussian_attn_layer());
        self.dncnn7A = nn.Sequential(*layers7A)

        layers8A.append(gaussian_attn_layer());
        self.dncnn8A = nn.Sequential(*layers8A)

        layers9A.append(gaussian_attn_layer());
        self.dncnn9A = nn.Sequential(*layers9A)

        layers10A.append(gaussian_attn_layer());
        self.dncnn10A = nn.Sequential(*layers10A)
        ##################################################################################################################
        layers1B.append(gaussian_attn_layer());
        self.dncnn1B = nn.Sequential(*layers1B)

        layers2B.append(gaussian_attn_layer());
        self.dncnn2B = nn.Sequential(*layers2B)

        layers3B.append(gaussian_attn_layer());
        self.dncnn3B = nn.Sequential(*layers3B)

        layers4B.append(gaussian_attn_layer());
        self.dncnn4B = nn.Sequential(*layers4B)

        layers5B.append(gaussian_attn_layer());
        self.dncnn5B = nn.Sequential(*layers5B)

        layers6B.append(gaussian_attn_layer());
        self.dncnn6B = nn.Sequential(*layers6B)

        layers7B.append(gaussian_attn_layer());
        self.dncnn7B = nn.Sequential(*layers7B)

        layers8B.append(gaussian_attn_layer());
        self.dncnn8B = nn.Sequential(*layers8B)

        layers9B.append(gaussian_attn_layer());
        self.dncnn9B = nn.Sequential(*layers9B)

        layers10B.append(gaussian_attn_layer());
        self.dncnn10B = nn.Sequential(*layers10B)

        layersEnd1=[nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size, padding=padding, bias=True)];
        self.dncnnEND1 = nn.Sequential(*layersEnd1)

        layersEnd2=[gaussian_attn_layer()];
        self.dncnnEND2 = nn.Sequential(*layersEnd2)

        layersEnd3=[gaussian_attn_layer()];
        self.dncnnEND3 = nn.Sequential(*layersEnd3)

        #######################################################################################################################
        self.lamb1 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb2 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb3 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb4 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb5 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb6 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb7 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb8 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb9 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb10 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb11 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb12 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb13 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb14 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb15 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb16 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb17 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb18 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb19 = Parameter(torch.ones(1), requires_grad=True)
        self.lamb20 = Parameter(torch.ones(1), requires_grad=True)

        #######################################################################################################################
        modules_head1 = [conv(3, 64, kernel_size=3, stride=1)]
        self.head1 = nn.Sequential(*modules_head1)
        modules_head2 = [conv(3, 64, kernel_size=3, stride=1)]
        self.head2 = nn.Sequential(*modules_head2)

        modules_head3 = [conv(3, 64, kernel_size=3, stride=1)]
        self.head3 = nn.Sequential(*modules_head3)
        modules_head4 = [conv(3, 64, kernel_size=3, stride=1)]
        self.head4 = nn.Sequential(*modules_head4)

        modules_tail = [conv(64, 3, kernel_size=3)]
        self.tail = nn.Sequential(*modules_tail)


    def forward(self,L, LW,L2M,H, HW,H2M ):
        Correct1=LW * L2M;         xA  = self.head3(Correct1)

        x1  = self.dncnn1(xA) ;    x1A = self.dncnn1A(self.head1(L) ); Add1 = x1*self.lamb1 + x1A;

        x2  = self.dncnn2(x1) ;    x2A = self.dncnn2A(Add1);           Add2 = x2*self.lamb2 + x2A;

        x3  = self.dncnn3(x2);     x3A = self.dncnn3A(Add2);           Add3 = x3*self.lamb3 + x3A;

        x4  = self.dncnn4(x3);     x4A = self.dncnn4A(Add3);           Add4 = x4*self.lamb4 + x4A;

        x5  = self.dncnn5(x4);     x5A = self.dncnn5A(Add4);           Add5 = x5*self.lamb5 + x5A;

        x6  = self.dncnn6(x5);     x6A = self.dncnn6A(Add5);           Add6 = x6*self.lamb6 + x6A;

        x7  = self.dncnn7(x6);     x7A = self.dncnn7A(Add6);           Add7 = x7*self.lamb7 + x7A;

        x8  = self.dncnn8(x7);     x8A = self.dncnn8A(Add7);           Add8 = x8*self.lamb8 + x8A;

        x9  = self.dncnn9(x8);     x9A = self.dncnn9A(Add8);           Add9 = x9*self.lamb9 + x9A;

        x10= self.dncnn10(x9);     x10A= self.dncnn10A(Add9);          Add10= x10*self.lamb10 + x10A;
        ################################################################################################
        Correct2=HW * H2M;            xB   = self.head4(Correct2)

        x11  = self.dncnn11(xB);      x1B = self.dncnn1B( self.head2(H) ); Add11 = x11*self.lamb11 + x1B;

        x12  = self.dncnn12(x11);     x2B = self.dncnn2B(Add11);           Add12 = x12*self.lamb12 + x2B;

        x13  = self.dncnn13(x12);     x3B = self.dncnn3B(Add12);           Add13 = x13*self.lamb13 + x3B;

        x14  = self.dncnn14(x13);     x4B = self.dncnn4B(Add13);           Add14 = x14*self.lamb14 + x4B;

        x15  = self.dncnn15(x14);     x5B = self.dncnn5B(Add14);           Add15 = x15*self.lamb15 + x5B;

        x16  = self.dncnn16(x15);     x6B = self.dncnn6B(Add15);           Add16 = x16*self.lamb16 + x6B;

        x17  = self.dncnn17(x16);     x7B = self.dncnn7B(Add16);           Add17 = x17*self.lamb17 + x7B;

        x18  = self.dncnn18(x17);     x8B = self.dncnn8B(Add17);           Add18 = x18*self.lamb18 + x8B;

        x19  = self.dncnn19(x18);     x9B = self.dncnn9B(Add18);           Add19 = x19*self.lamb19 + x9B;

        x20  = self.dncnn20(x19);     x10B= self.dncnn10B(Add19);          Add20 = x20*self.lamb20 + x10B;

        y1   = self.tail(self.dncnnEND3(self.dncnnEND2(self.dncnnEND1( torch.cat([Add10,Add20],dim=1) )) )) + (Correct1 + Correct2)
        out =torch.clamp(y1, 0.0, 1.0)

        return out
##################################################################################################################################


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G1(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    return init_net(net, init_type, init_gain, gpu_id)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)

        self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)

        self.outc = Outconv(ngf, output_nc)
    #

    def forward(self, input):
        out = {}
        out['in'] = self.inc(input)
        out['d1'] = self.down1(out['in'])
        out['d2'] = self.down2(out['d1'])
        out['bottle'] = self.resblocks(out['d2'])
        out['u1'] = self.up1(out['bottle'])
        out['u2'] = self.up2(out['u1'])
        return self.outc(out['u2'])

    # def forward(self, input):
    #     out = {}
    #     out['in'] = self.inc(input)
    #     out['d1'] = self.down1(out['in'])
    #     out['d2'] = self.down2(out['d1'])
    #
    #     out['bottle'] = self.resblocks(out['d2'])
    #
    #     out['u1'] = self.up1(out['bottle'])+out['d1']
    #     out['u2'] = self.up2(out['u1'])+out['in']
    #
    #     return self.outc(out['u2'])+input


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                      bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(  nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=use_bias),  norm_layer(out_ch),  nn.ReLU(True)  )

    def forward(self, x):
        x = self.down(x)
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim),  nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)

# Define a Resnet block
class ResBlock2(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock2, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim),  nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)



class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential( nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential( nn.ReflectionPad2d(3), nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, x):
        x = self.outconv(x)
        return x


def define_G2(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
              gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = ResnetGenerator2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    return init_net(net, init_type, init_gain, gpu_id)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
