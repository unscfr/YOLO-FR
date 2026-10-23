import torch
import torch.nn as nn


class PAT_CH(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        # 使用分组卷积处理各通道的均值和方差
        self.conv = nn.Conv2d(2*inchannels, inchannels, kernel_size=1, groups=inchannels)
        self.cv = nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1)
        self.cv2 = nn.Conv2d(2*inchannels, inchannels, kernel_size=1, groups=inchannels)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(inchannels)

    def forward(self, x):
        # 保持维度计算均值和方差
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True)# + 1e-6  # 防止除零
        # 拼接空间统计量并计算通道注意力
        mv = torch.cat([mean, var], dim=1)
        weights = self.conv(mv)
        weights = self.sigmoid(weights)

        out = x * weights
        out = self.bn(out)

        x1 = self.cv(x)
        out = torch.cat([out, x1], dim=1)

        return self.cv2(out)


if __name__ == "__main__":
    # 输入大小: [batch_size, channels, height, width]
    x = torch.randn(16, 32, 8, 8)  # 64通道，32x32尺寸的输入

    model = PAT_CH(32, 32)
    output = model(x)

    print(f"Output shape: {output.shape}")  # 输出形状
