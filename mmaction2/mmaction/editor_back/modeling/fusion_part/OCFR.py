import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# Cut & paste from PyTorch official master until it's in a few official releases - RW
# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

class OCFR(nn.Module):
    def __init__(self, dim=768, num_class=12, momentum=0.8, alpha=1.0, beta=1.0, temp=0.05):
        super(OCFR, self).__init__()
        self.dim = dim
        self.RGB_centers = nn.Parameter(torch.zeros(num_class, self.dim), requires_grad=False)
        self.NIR_centers = nn.Parameter(torch.zeros(num_class, self.dim), requires_grad=False)
        self.momentum = torch.tensor(momentum, dtype=torch.float32)
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

    def compute_center(self, features):
        # 计算所有特征的均值作为中心
        center = features.mean(dim=0)
        return center.detach().unsqueeze(0)  # 需要将结果维度调整为 (1, dim)

    def compute_intra_loss(self, center, features):
        centers_modality = center.repeat(features.size(0), 1)
        loss = nn.MSELoss()(centers_modality, features)
        return loss

    def forward(self, RGB_feat, NIR_feat):
        # 归一化特征
        RGB_feat = F.normalize(RGB_feat, dim=1)
        NIR_feat = F.normalize(NIR_feat, dim=1)


        # 更新中心
        self.update(RGB_feat, NIR_feat)

        # 计算损失
        RGB_center = self.RGB_centers[0]  # 只有一个类别
        NIR_center = self.NIR_centers[0]  # 只有一个类别

        intra_loss = (self.compute_intra_loss(RGB_center, RGB_feat) +
                      self.compute_intra_loss(NIR_center, NIR_feat))

        total_intra_loss = self.alpha * intra_loss
        return total_intra_loss

    def update(self, RGB_feat, NIR_feat):
        RGB_center = self.compute_center(RGB_feat)
        NIR_center = self.compute_center(NIR_feat)

        # 更新类别中心
        self.RGB_centers[0] = self.momentum * RGB_center + (1 - self.momentum) * self.RGB_centers[0]
        self.NIR_centers[0] = self.momentum * NIR_center + (1 - self.momentum) * self.NIR_centers[0]
