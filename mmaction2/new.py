import torch
import torch.nn as nn
from functools import partial

NORM_EPS = 1e-5  # 设置一个小的常数来避免除零错误

class MaskedMHCA(nn.Module):
    """
    Masked Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels, head_dim):
        super(MaskedMHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.head_dim = head_dim
        self.num_heads = out_channels // head_dim
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=self.num_heads, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, mask):
        # x: 输入特征图，形状为 (B, C, H, W)
        # mask: 掩码，形状为 (B, 1, H, W)

        # 进行卷积
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)

        # 应用掩码
        out = out * mask

        # 投影
        out = self.projection(out)
        return out

if __name__=='__main__':
    model=MaskedMHCA(out_channels=768,head_dim=64)
    x=torch.randn(1,3137,768)
    mask=torch.randn(1,3137,1)
    ouy=model(x,mask)
    print(ouy.shape)