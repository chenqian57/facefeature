import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

from nets.iresnet import (iresnet18, iresnet34, iresnet50, iresnet100,
                          iresnet200)
from nets.mobilefacenet import get_mbf
from nets.mobilenet import get_mobilenet

from nets.resnet import (ResNet, resnet18, resnet34, resnet50, resnet101,
           resnet152, resnext50_32x4d, resnext101_32x8d,
           wide_resnet50_2, wide_resnet101_2)

from nets.iresnet2060 import iresnet2060
from nets.vit import get_vit



# def l2_norm(input, axis=1):
#     norm = torch.norm(input, 2, axis, True)
#     output = torch.div(input, norm)
#     return output




# class Arcface_Head(Module):
#     # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
#     def __init__(self, embedding_size=128, num_classes=144608, s=64., m=0.5):
#         super(Arcface_Head, self).__init__()
#         self.classnum = num_classes
#         self.kernel = Parameter(torch.Tensor(embedding_size, num_classes))
#         # initial kernel
#         self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
#         # self.kernel2.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         self.m = m  # the margin value, default is 0.5
#         self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.mm = self.sin_m * m  # issue 1
#         self.threshold = math.cos(math.pi - m)

#     def forward(self, embbedings, label):
#         # weights norm
#         nB = len(embbedings)
#         kernel_norm = l2_norm(self.kernel, axis=0)
#         # kernel_norm2 = l2_norm(self.kernel2,axis=0)
#         # cos(theta+m)
#         cos_theta = torch.mm(embbedings, kernel_norm)
#         cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability   为了数值稳定性

#         cos_theta_2 = torch.pow(cos_theta, 2)
#         sin_theta_2 = 1 - cos_theta_2

#         # 计算出 cos_theta 的平方 cos_theta_2 和 sin_theta 的平方 sin_theta_2


#         sin_theta = torch.sqrt(sin_theta_2)
#         # 使用 torch.sqrt 方法计算出 sin_theta 张量
#         cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

#         cond_v = cos_theta - self.threshold
#         cond_mask = cond_v <= 0
#         keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
#         cos_theta_m[cond_mask] = keep_val[cond_mask]
#         output = cos_theta * 1.0  

        
#         idx_ = torch.arange(0, nB, dtype=torch.long)
#         output[idx_, label] = cos_theta_m[idx_, label]

#         output *= self.s  # scale up in order to make softmax work, first introduced in normface

#         return output





















class Arcface_Head(Module):
    def __init__(self, embedding_size=128, num_classes=10575, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine  = F.linear(input, F.normalize(self.weight))
        sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))


        # phi = torch.where(label != -1)[0]


        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output  *= self.s
        return output






def get_model(name, **kwargs):
    # resnet
    if name == "ir18":
        return iresnet18(False, **kwargs)
    elif name == "ir34":
        return iresnet34(False, **kwargs)
    elif name == "ir50":
        return iresnet50(False, **kwargs)
    elif name == "ir100":
        return iresnet100(False, **kwargs)
    elif name == "ir200":
        return iresnet200(False, **kwargs)
    else:
        raise ValueError()
















class Arcface(nn.Module):
    def __init__(self, num_classes=None, backbone="mobilefacenet", pretrained=False, mode="train"):
        super(Arcface, self).__init__()
        if backbone=="mobilefacenet":
            embedding_size  = 128
            s               = 32
            self.arcface    = get_mbf(embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="mobilenetv1":
            embedding_size  = 512
            s               = 64
            self.arcface    = get_mobilenet(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet18":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet18(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet34":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet34(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)


        elif backbone=="iresnet50":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet50(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)


        # elif backbone=="iresnet50":
        #     embedding_size  = 512
        #     s               = 64
        #     self.arcface    = iresnet50(False, **kwargs)



        elif backbone=="iresnet100":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet100(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet200":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet200(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)



        elif backbone=="iresnet2060":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet2060(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)





        elif backbone=="vit":
            embedding_size  = 128
            s               = 32
            self.arcface    = get_vit(embedding_size=embedding_size, pretrained=pretrained)







        elif backbone=="resnet34":
            embedding_size  = 512
            s               = 64
            self.arcface    = resnet34( dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)
            # dropout_keep_prob=0.5,




        elif backbone=="resnet50":
            embedding_size  = 512
            s               = 64
            self.arcface    = resnet50( dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)
            # dropout_keep_prob=0.5,




        elif backbone=="resnet101":
            embedding_size  = 512
            s               = 64
            self.arcface    = resnet101( dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)
            # dropout_keep_prob=0.5,



        elif backbone=="resnet152":
            embedding_size  = 512
            s               = 64
            self.arcface    = resnet152( dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)
            # dropout_keep_prob=0.5,
        


        # elif backbone=="iresnet100":
        #     embedding_size  = 512
        #     s               = 64
        #     self.arcface    = iresnet100(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        



        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1.'.format(backbone))

        self.mode = mode
        if mode == "train":
            self.head = Arcface_Head(embedding_size=embedding_size, num_classes=num_classes, s=s)






    def forward(self, x, y = None, mode = "predict"):
        x = self.arcface(x)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x)
        if mode == "predict":
            return x
        else:
            x = self.head(x, y)
            return x
