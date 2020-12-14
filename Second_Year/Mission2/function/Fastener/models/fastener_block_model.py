import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        inplanes = 256
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# http://d2l.ai/chapter_convolutional-modern/resnet.html
class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        """ Upscale - Cat - Conv """
        x1 = self.up(x1)
        diffY = torch.tensor([x1.size()[2] - x2.size()[2]])
        diffX = torch.tensor([x1.size()[3] - x2.size()[3]])
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Out(nn.Module):
    """ Last block """
    def __init__(self, input_channels, mid_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        """ Upscale - Conv """
        x = self.up(x)
        return self.conv(x)


class FastenerBlockModel(nn.Module):
    def __init__(self, in_channels = 3, out_channels_id = 5):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(ASPP())

        self.u1_id = Up(256 + 128, 128)
        self.u2_id = Up(128 + 64, 64)
        self.u3_id = Up(64 + 64, 64)
        self.o_id = Out(64, 64, out_channels_id)

        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, self.u1_id, self.u2_id, self.u3_id, self.o_id)

    def forward(self, x):
        # back-bone
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)

        # id
        id = self.u1_id(x5, x3)
        id = self.u2_id(id, x2)
        id = self.u3_id(id, x1)
        id = self.o_id(id)

        return id


if __name__ == '__main__':
    # check shapes
    X = torch.rand(size=(1, 3, 320, 240))
    print('input shape : ', X.shape)
    
    model = CorrespondenceBlockModel()
    net = model.net

    x1 = X = net[0](X)
    x2 = X = net[1](X)
    x3 = X = net[2](X)
    x4 = X = net[3](X)
    x5 = X = net[4](X)
    y1 = X = net[5](X, x3)
    y2 = X = net[6](X, x2)
    y3 = X = net[7](X, x1)
    y4 = X = net[8](X)

    outputs = [x1, x2, x3, x4, x5, y1, y2, y3, y4]
    for layer, output in zip(net, outputs):
        print(layer.__class__.__name__, 'output shape:\t', output.shape)