from torch import nn
import torch
from .common_utils_mbyolo import *


from ultralytics.nn.modules import C2f, Bottleneck, Conv, C3
from ultralytics.nn.modules.Myconv import WTConv2d
from .mamba_yolo import SS2D


class C3k_WT_Mamba(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_WT_Mamba(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_WT_Mamba(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_WT_Mamba(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WT_Mamba(self.c, self.c, shortcut, g) for _ in range(n)
        )

class Bottleneck_WT_Mamba(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,
                 in_channels: int = 0,
                 hidden_dim: int = 0,
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
                 # =============================
                 ssm_d_state: int = 16,
                 ssm_ratio=2.0,
                 ssm_rank_ratio=2.0,
                 ssm_dt_rank: Any = "auto",
                 ssm_act_layer=nn.SiLU,
                 ssm_conv: int = 3,
                 ssm_conv_bias=True,
                 ssm_drop_rate: float = 0,
                 ssm_init="v0",
                 forward_type="v2",
                 # =============================
                 mlp_ratio=4.0,
                 mlp_act_layer=nn.GELU,
                 mlp_drop_rate: float = 0.0,
                 # =============================
                 use_checkpoint: bool = False,
                 post_norm: bool = False,
                 **kwargs,
                 ):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        print(c_, c2)
        self.cv2 = WTConv2d(c_, c2)
        self.mamba = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.ss2d(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))