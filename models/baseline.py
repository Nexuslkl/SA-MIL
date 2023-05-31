import torch
import torch.nn as nn
import torchvision


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.stage1 = nn.Sequential(
            Conv(3, 64),
            Conv(64, 64)
        )
        self.stage2 = nn.Sequential(
            Conv(64, 128),
            Conv(128, 128)
        )
        self.stage3 = nn.Sequential(
            Conv(128, 256),
            Conv(256, 256),
            Conv(256, 256)
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )

    def pretrain(self):
        model_pre = torchvision.models.vgg16(pretrained=True)
        self.stage1[0].conv[0] = model_pre.features[0]
        self.stage1[1].conv[0] = model_pre.features[2]
        self.stage2[0].conv[0] = model_pre.features[5]
        self.stage2[1].conv[0] = model_pre.features[7]
        self.stage3[0].conv[0] = model_pre.features[10]
        self.stage3[1].conv[0] = model_pre.features[12]
        self.stage3[2].conv[0] = model_pre.features[14]

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.decoder(x)

        return x


