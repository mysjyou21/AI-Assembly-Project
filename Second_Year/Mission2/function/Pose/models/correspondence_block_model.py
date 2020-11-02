import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
        """ Upscale - Stack - Conv """
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
        """ Upscale - Stack - Conv """
        x = self.up(x)
        return self.conv(x)


class CorrespondenceBlockModel(nn.Module):
    def __init__(self, in_channels = 3, out_channels_id = 7, out_channels_uvw = 256):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))

        self.u1_id = Up(256 + 128, 128)
        self.u2_id = Up(128 + 64, 64)
        self.u3_id = Up(64 + 64, 64)
        self.o_id = Out(64, 64, out_channels_id)

        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.u1_id, self.u2_id, self.u3_id, self.o_id)

        self.u1_u = Up(256 + 128, 128)
        self.u2_u = Up(128 + 64, 64)
        self.u3_u = Up(64 + 64, 64)
        self.o_u = Out(64, 64, out_channels_uvw)

        self.u1_v = Up(256 + 128, 128)
        self.u2_v = Up(128 + 64, 64)
        self.u3_v = Up(64 + 64, 64)
        self.o_v = Out(64, 64, out_channels_uvw)

        self.u1_w = Up(256 + 128, 128)
        self.u2_w = Up(128 + 64, 64)
        self.u3_w = Up(64 + 64, 64)
        self.o_w = Out(64, 64, out_channels_uvw)

    def forward(self, x):
        # back-bone
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)

        # id
        id = self.u1_id(x4, x3)
        id = self.u2_id(id, x2)
        id = self.u3_id(id, x1)
        id = self.o_id(id)

        # u
        u = self.u1_u(x4, x3)
        u = self.u2_u(u, x2)
        u = self.u3_u(u, x1)
        u = self.o_u(u)

        # v
        v = self.u1_v(x4, x3)
        v = self.u2_v(v, x2)
        v = self.u3_v(v, x1)
        v = self.o_v(v)

        # w
        w = self.u1_w(x4, x3)
        w = self.u2_w(w, x2)
        w = self.u3_w(w, x1)
        w = self.o_w(w)

        return id, u, v, w


if __name__ == '__main__':
    # check shapes
    X = torch.rand(size=(1, 3, 320, 240))
    model = CorrespondenceBlockModel()
    net = model.net

    print('input shape : ', X.shape)
    x1 = X = net[0](X)
    print(net[0].__class__.__name__, 'output shape:\t', X.shape)
    x2 = X = net[1](X)
    x3 = X = net[2](X)
    x4 = X = net[3](X)
    y1 = X = net[4](X, x3)
    y2 = X = net[5](X, x2)
    y3 = X = net[6](X, x1)
    y4 = X = net[7](X)

    outputs = [x1, x2, x3, x4, y1, y2, y3, y4]
    for layer, output in zip(net, outputs):
        print(layer.__class__.__name__, 'output shape:\t', output.shape)
