"""
From https://github.com/mit-han-lab/e3d/blob/db65d6c968a12d30a4caa2715ef6766ec04d7505/spvnas/core/models/semantic_kitti/minkunet.py
"""

import time
from collections import OrderedDict
import torch
import torchsparse
import torch.nn as nn
import torchsparse.nn as spnn
import MinkowskiEngine as ME
import numpy as np

__all__ = ['MinkUNet']

# _ks = 7
_ks = 3


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        if stride == 1 and ks == 3:
            ks = _ks
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicConvolutionBlockdown(nn.Module):
    def __init__(self, inc, outc, ks=2, stride=2, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            # spnn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        if stride == 1 and ks == 3:
            ks = _ks
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transpose=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlockup(nn.Module):
    def __init__(self, inc, outc, ks=2, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transpose=True),
            spnn.BatchNorm(outc),
            # spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


# 2023-10-22 Jinzheng Guang resnet
class ResidualPath(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            # spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, resblock_att_status=False):
        super().__init__()
        if stride == 1 and ks == 3:
            ks = _ks
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc)
        )
        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )
        self.relu = spnn.ReLU(True)
        self.resblock_att = MobileViTv2Attention(d_model=outc)
        self.resblock_att_status = resblock_att_status

    def forward(self, x):
        if self.resblock_att_status:
            out = self.relu(self.resblock_att(self.net(x)) + self.downsample(x))
        else:
            out = self.relu(self.net(x) + self.downsample(x))
        return out


# 2023-10-15 Jinzheng Guang MobileViTv2Attention
# Separable Self-attention for Mobile Vision Transformers
class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        inputs = torch.randn(50000, 512)
        sa = MobileViTv2Attention(d_model=512)
        output = sa(inputs)
        print(output.shape)
        '''
        input = x.F

        i = self.fc_i(input)  # input (n, c)  i (n, 1)   # (bs,nq,1)
        weight_i = torch.softmax(i, dim=0)  # (n, 1) # bs,nq,1
        context_score = weight_i * self.fc_k(input)  # (n, c) # bs,nq,d_model
        context_vector = torch.sum(context_score, dim=0, keepdim=True)  # (1, c) # bs,1,d_model
        v = self.fc_v(input) * context_vector  # (n, c) # bs,nq,d_model
        out = self.fc_o(v)  # (n, c) # bs,nq,d_model

        x.F = out
        return x


class MinkUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        input_dim = kwargs.get("input_dim", 3)

        self.stem = nn.Sequential(
            # spnn.Conv3d(3, cs[0], kernel_size=3, stride=1),
            spnn.Conv3d(input_dim, cs[0], kernel_size=_ks, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=_ks, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True)
        )
        resblock_att_status = kwargs.get('resblock', True)
        assert isinstance(resblock_att_status, bool)
        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, resblock_att_status=resblock_att_status),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, resblock_att_status=resblock_att_status)
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, resblock_att_status=resblock_att_status),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, resblock_att_status=resblock_att_status),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        # 2023-10-03 Jinzheng Guang
        # cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        self.path_status = kwargs.get('path')
        self.pathvit_att_status = kwargs.get('pathatt')
        assert isinstance(self.path_status, bool)
        assert isinstance(self.pathvit_att_status, bool)

        # 2023-10-07 Jinzheng Guang
        self.pathx3 = nn.ModuleList([
            ResidualPath(cs[0], cs[0]),
            ResidualPath(cs[0], cs[0]),
            ResidualPath(cs[0], cs[0]),
        ])
        self.upx3 = nn.ModuleList([
            BasicDeconvolutionBlockup(cs[1], cs[0]),
            BasicDeconvolutionBlockup(cs[1], cs[0]),
            BasicDeconvolutionBlockup(cs[1], cs[0]),
        ])
        self.downx3 = nn.ModuleList([
            BasicConvolutionBlockdown(cs[0], cs[0]),
            BasicConvolutionBlockdown(cs[0], cs[0]),
            BasicConvolutionBlockdown(cs[0], cs[0]),
        ])

        self.pathx2 = nn.ModuleList([
            ResidualPath(cs[1], cs[1]),
            ResidualPath(cs[1], cs[1]),
            ResidualPath(cs[1], cs[1]),
        ])
        self.upx2 = nn.ModuleList([
            BasicDeconvolutionBlockup(cs[2], cs[1]),
            BasicDeconvolutionBlockup(cs[2], cs[1]),
        ])
        self.downx2 = nn.ModuleList([
            BasicConvolutionBlockdown(cs[1], cs[2]),
            BasicConvolutionBlockdown(cs[1], cs[2]),
        ])

        self.pathx1 = nn.ModuleList([
            ResidualPath(cs[2], cs[2]),
            ResidualPath(cs[2], cs[2]),
        ])
        self.upx1 = BasicDeconvolutionBlockup(cs[3], cs[2])
        self.downx1 = BasicConvolutionBlockdown(cs[2], cs[3])

        self.pathx0 = ResidualPath(cs[3], cs[3])
        self.relux = nn.ModuleList([
            spnn.ReLU(True), spnn.ReLU(True), spnn.ReLU(True),
            spnn.ReLU(True), spnn.ReLU(True), spnn.ReLU(True),
            spnn.ReLU(True), spnn.ReLU(True), spnn.ReLU(True),
        ])

        # 2023-10-08 Jinzheng Guang
        self.path_res_status = kwargs.get('pathres')
        assert isinstance(self.path_res_status, bool)
        self.path_cat = nn.ModuleList([
            ResidualPath(cs[3], cs[3]//2), ResidualPath(cs[3], cs[3]//2),
            ResidualPath(cs[2], cs[2]//2), ResidualPath(cs[2], cs[2]//2),
            ResidualPath(cs[1], cs[1]//2), ResidualPath(cs[1], cs[1]//2),
            ResidualPath(cs[0], cs[0]//2), ResidualPath(cs[0], cs[0]//2),
        ])

        self.pathvit_att = nn.ModuleList([
            MobileViTv2Attention(d_model=cs[3]),
            MobileViTv2Attention(d_model=cs[2]),
            MobileViTv2Attention(d_model=cs[1]),
            MobileViTv2Attention(d_model=cs[0]),
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, spnn.Conv3d):
                # ME 的何凯明初始化
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, spnn.BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, std=0.01)
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)
            else:
                pass

    def _forward_paths_res(self, px3, px2, px1, px0):
        # CVPR2018: Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment
        pathx30_out = self.relux[0](self.upx3[0](px2) + self.pathx3[0](px3))
        pathx20_out = self.relux[1](self.upx2[0](px1) + self.pathx2[0](px2) + self.downx3[0](pathx30_out))
        pathx1_out = self.relux[2](self.upx1(px0) + self.pathx1[0](px1) + self.downx2[0](pathx20_out))
        if self.pathvit_att_status:
            px0_out = self.relux[3](px0 + self.pathvit_att[0](self.pathx0(px0)) + self.downx1(pathx1_out))
        else:
            px0_out = self.relux[3](px0 + self.pathx0(px0) + self.downx1(pathx1_out))

        pathx31_out = self.relux[4](self.upx3[1](pathx20_out) + self.pathx3[1](pathx30_out))
        pathx21_out = self.relux[5](self.upx2[1](pathx1_out) + self.pathx2[1](pathx20_out) + self.downx3[1](pathx31_out))
        if self.pathvit_att_status:
            px1_out = self.relux[6](px1 + self.pathvit_att[1](self.pathx1[1](pathx1_out)) + self.downx2[1](pathx21_out))
            px3_out = self.relux[7](px3 + self.upx3[2](pathx21_out) + self.pathvit_att[3](self.pathx3[2](pathx31_out)))
            px2_out = self.relux[8](px2 + self.pathvit_att[2](self.pathx2[2](pathx21_out)) + self.downx3[2](px3_out))
        else:
            px1_out = self.relux[6](px1 + self.pathx1[1](pathx1_out) + self.downx2[1](pathx21_out))
            px3_out = self.relux[7](px3 + self.upx3[2](pathx21_out) + self.pathx3[2](pathx31_out))
            px2_out = self.relux[8](px2 + self.pathx2[2](pathx21_out) + self.downx3[2](px3_out))

        return px3_out, px2_out, px1_out, px0_out

    def _forward_paths_cat(self, px3, px2, px1, px0):
        pathx30_out = self.relux[0](self.upx3[0](px2) + self.pathx3[0](px3))
        pathx20_out = self.relux[1](self.upx2[0](px1) + self.pathx2[0](px2) + self.downx3[0](pathx30_out))
        pathx1_out = self.relux[2](self.upx1(px0) + self.pathx1[0](px1) + self.downx2[0](pathx20_out))
        if self.pathvit_att_status:
            px0_out = torchsparse.cat([self.path_cat[0](px0), self.path_cat[1](self.relux[3](self.pathvit_att[0](self.pathx0(px0)) + self.downx1(pathx1_out)))])
        else:
            px0_out = torchsparse.cat([self.path_cat[0](px0), self.path_cat[1](self.relux[3](                    self.pathx0(px0)  + self.downx1(pathx1_out)))])

        pathx31_out = self.relux[4](self.upx3[1](pathx20_out) + self.pathx3[1](pathx30_out))
        pathx21_out = self.relux[5](self.upx2[1](pathx1_out) + self.pathx2[1](pathx20_out) + self.downx3[1](pathx31_out))
        if self.pathvit_att_status:
            px1_out = torchsparse.cat([self.path_cat[2](px1), self.path_cat[3](self.relux[6](self.pathvit_att[1](self.pathx1[1](pathx1_out)) + self.downx2[1](pathx21_out)))])
            px3_out = torchsparse.cat([self.path_cat[4](px3), self.path_cat[5](self.relux[7](self.pathvit_att[3](self.pathx3[2](pathx31_out)) + self.upx3[2](pathx21_out)))])
            px2_out = torchsparse.cat([self.path_cat[6](px2), self.path_cat[7](self.relux[8](self.pathvit_att[2](self.pathx2[2](pathx21_out)) + self.downx3[2](px3_out)))])
        else:
            px1_out = torchsparse.cat([self.path_cat[2](px1), self.path_cat[3](self.relux[6](                     self.pathx1[1](pathx1_out) + self.downx2[1](pathx21_out)))])
            px3_out = torchsparse.cat([self.path_cat[4](px3), self.path_cat[5](self.relux[7](                     self.pathx3[2](pathx31_out) + self.upx3[2](pathx21_out)))])
            px2_out = torchsparse.cat([self.path_cat[6](px2), self.path_cat[7](self.relux[8](                     self.pathx2[2](pathx21_out) + self.downx3[2](px3_out)))])

        return px3_out, px2_out, px1_out, px0_out

    def forward(self, x):
        x0 = self.stem(x)  # 621,687 x 3
        x1 = self.stage1(x0)  # 621,687 x 32
        x2 = self.stage2(x1)  # 362,687 x 32
        x3 = self.stage3(x2)  # 192,434 x 64
        x4 = self.stage4(x3)  # 94,584 x 128

        if self.path_status:
            if self.path_res_status:
                x0, x1, x2, x3 = self._forward_paths_res(x0, x1, x2, x3)
            else:
                x0, x1, x2, x3 = self._forward_paths_cat(x0, x1, x2, x3)

        y1 = self.up1[0](x4)  # 42,187 x 256
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)  # 94,584 x 256

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)  # 192,434 x 128

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)  # 362,687 x 96

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)  # 621,687 x 96

        out = self.classifier(y4.F)  # 621,687 x 31

        return out  # (n, 31)

if __name__ == '__main__':
    # kwargs1 = {'cr': 1.0, 'input_dim': 3, 'num_classes': 31, 'path': True, 'pathatt': True, 'resblock': True, 'run_up': True}
    kwargs = {'path': True, 'pathatt': True, 'resblock': True}
    models = MinkUNet(cr=1.0, run_up=True, num_classes=31, input_dim=3, **kwargs)

    coords = torch.randn(2000, 4)
    feats = torch.randn(2000, 3)
    inputs = torchsparse.SparseTensor(coords=coords, feats=feats)

    output = models(inputs)
    print(output.shape)

# 2023-10-03 Jinzheng Guang
