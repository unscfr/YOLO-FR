import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.new.utils import ConvLayer1D, ConvLayer2D, LayerNorm1D, FFN


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3 * state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = ConvLayer1D(d_model, 2 * self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        batch, _, L = x.shape
        H = int(math.sqrt(L))

        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, H)).flatten(2)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)

        AB = (A * B)
        h = x @ AB.transpose(-2, -1)

        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ C  # B C N, B C L -> B C L

        y = y.view(batch, -1, H, H).contiguous()  # + x * self.D  # B C H W
        return y, h


class EfficientViMBlock(nn.Module):
    def __init__(self, indim, outdim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()

        self.mlp_ratio = mlp_ratio

        self.mixer = HSMSSD(d_model=indim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(indim)

        self.dwconv1 = ConvLayer2D(indim, indim, 3, padding=1, groups=indim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(indim, indim//2, 3, padding=1, groups=indim//2, bn_weight_init=0, act_layer=None)

        self.ffn = FFN(in_dim=indim//2, dim=int(indim//2 * mlp_ratio))

        # LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, indim), requires_grad=True)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # DWconv1
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        # HSM-SSD
        x_prev = x
        x, h = self.mixer(self.norm(x.flatten(2)))
        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        # DWConv2
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)

        # FFN
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        #print(x.shape)
        return x

# 测试模块
if __name__ == "__main__":
    # 假设输入为(batch_size, in_channels, height, width)
    input_tensor = torch.randn(3, 64, 8, 8)  # Batch size 8, 输入通道64，32x32的特征图
    model = EfficientViMBlock(64)
    output_tensor = model(input_tensor)
    print(f"Output shape: {output_tensor.shape}")  # 输出形状
