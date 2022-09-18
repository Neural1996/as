import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.convnext import convnext_tiny
from lib.convnext import LayerNorm
"""
此文件是构建整体网络，包含各种网络模块
"""

class Pred_Layer(nn.Module):
    def __init__(self, in_c=32):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x


class BAM(nn.Module):
    def __init__(self, in_c):
        super(BAM, self).__init__()
        self.reduce = nn.Conv2d(in_c, 32, 1)  # 1×1 对rgb和深度特征合并后进行降维
        self.ff_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.bf_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgb_pred_layer = Pred_Layer(32 * 2)

    def forward(self, rgb_feat, pred):
        feat = rgb_feat
        feat = self.reduce(feat)   # 得到的是original feature
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(      # 得到FF attention map
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)  # 正向
        bf_feat = self.bf_conv(feat * (1 - pred))  # 反向
        new_pred = self.rgb_pred_layer(torch.cat((ff_feat, bf_feat), 1))  # 得到residual
        return new_pred

# ASPP for MBAM
class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

# MBAM
class MBAM(nn.Module):
    def __init__(self, in_c):
        super(MBAM, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgb_pred_layer = Pred_Layer(32 * 8)

    def forward(self, rgb_feat, pred):
        feat = rgb_feat
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        new_pred = self.rgb_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        return new_pred


class Network(nn.Module):
    # convnext based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- Backbone ----
        self.convnext = convnext_tiny(pretrained=imagenet_pretrained)
        # rgb6
        self.toplayer = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(768, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        # Global Pred
        self.rgb_global = Pred_Layer(32)

        # Shor-Conection
        self.bams = nn.ModuleList([        # 中间层
            BAM(96),  # 输入的维度
            BAM(96),
            MBAM(192),
            MBAM(384),
            MBAM(768),
        ])

        self.layernorm1 = LayerNorm(96, eps=1e-6, data_format="channels_first")
        self.layernorm2 = LayerNorm(192, eps=1e-6, data_format="channels_first")
        self.layernorm3 = LayerNorm(384, eps=1e-6, data_format="channels_first")
        self.layernorm4 = LayerNorm(768, eps=1e-6, data_format="channels_first")

    def _upsample_add(self, x, y):         # 上采样之后add，BAM中最后使用
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        [_, _, H, W] = x.size()
        # Feature Extraction
        x_0 = self.convnext.downsample_layers[0](x)  # stem                       bs, 96, 56, 56  自带layernorm
        x1 = self.convnext.stages[0](x_0)            # 进入第一个stage
        x_1 = self.layernorm1(x1)                           # 第一个                     bs, 96, 56, 56
        x2 = self.convnext.stages[1](self.convnext.downsample_layers[1](x1))
        x_2 = self.layernorm2(x2)                                                      # bs, 192, 28, 28
        x3 = self.convnext.stages[2](self.convnext.downsample_layers[2](x2))
        x_3 = self.layernorm3(x3)                                                      # bs, 384, 14, 14
        x4 = self.convnext.stages[3](self.convnext.downsample_layers[3](x3))
        x_4 = self.layernorm4(x4)                                                      # bs, 768, 7, 7
        x_5 = self.toplayer(x4)
        feats = [x_0, x_1, x_2, x_3, x_4, x_5]  # 6个输出

        s6_pred = self.rgb_global(x_5)  # 生成s6预测

        preds = [
                F.interpolate(s6_pred,
                              size=(H, W),
                              mode='bilinear',
                              align_corners=True)
        ]   # 记录S6生成的预测

        p = s6_pred

        for idx in [4, 3, 2, 1, 0]:
            _p = self.bams[idx](feats[idx], p)  # _p为生成的FF与BF concat后sigmoid
            p = self._upsample_add(p, _p)   # p为真正的新预测
            preds.append(
                    F.interpolate(p,
                                  size=(H, W),
                                  mode='bilinear',
                                  align_corners=True))   # 对预测进行记录

        predict6 = preds[0]  # s6
        predict5 = preds[1]
        predict4 = preds[2]
        predict3 = preds[3]
        predict2 = preds[4]
        predict1 = preds[5]

        return torch.sigmoid(predict6), torch.sigmoid(predict5), torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(
            predict1)


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(10):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y)