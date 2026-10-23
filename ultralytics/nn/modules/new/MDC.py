
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, DWConv


class MDC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 两遍卷积
        self.cv1 = Conv(in_channels, in_channels//2, 1, 1)
        self.cv2 = Conv(in_channels//2, in_channels//2, 3, 2, 1)
        self.cv3 = Conv(in_channels, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dwcv1 = DWConv(in_channels//4, in_channels//4, s=2)

        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.bn2 = nn.BatchNorm2d(in_channels//2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.silu(self.bn1(self.cv1(x)))
        x2 = x
        channels = x.size(1)
        x1, x3 = torch.split(x, [channels // 2, channels // 2], dim=1)
        x1 = self.maxpool(x1)
        x2 = self.silu(self.bn2(self.cv2(x2)))
        x3 = self.dwcv1(x3)
        #print(x1.size(), x2.size(), x3.size())
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.silu(self.bn3(self.cv3(x)))

        return x