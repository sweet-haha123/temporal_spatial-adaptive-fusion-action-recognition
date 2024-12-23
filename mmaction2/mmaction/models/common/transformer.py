# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils import digit_version

from mmaction.registry import MODELS
# from mmengine.registry import MODELS
import torch.nn.functional as F
import torchvision.utils as vutils
import cv2
import numpy as np

from mmengine.visualization import TensorboardVisBackend

caculate_pic=0
caculate_pic_batch=1
temporal_attn=None
spatial_attn=None
# 创建一个SummaryWriter
writer = TensorboardVisBackend(save_dir='/home/qingyuhan/code/mmaction2/1')
class Multihead_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


@MODELS.register_module()
class DividedTemporalAttentionWithNorm(BaseModule):
    """Temporal Attention in Divided Space Time Attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Defaults to 0..
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
            Defaults to `dict(type='DropPath', drop_prob=0.1)`.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
        init_cfg (dict | None): The Config for initialization. Defaults to
            None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)

        # #add by qyh
        # self.temporal_conv=nn.Conv1d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=3, stride=1, padding=1)
        # #end


        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        constant_init(self.temporal_fc, val=0, bias=0)

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        """Defines the computation performed at every call."""
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        init_cls_token = query[:, 0, :].unsqueeze(1)
        identity = query_t = query[:, 1:, :]

        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // self.num_frames,  self.num_frames

        # res_temporal [batch_size * num_patches, num_frames, embed_dims]
        query_t = self.norm(query_t.reshape(b * p, t, m)).permute(1, 0, 2)
        res_temporal, res_temporal_attn = self.attn(query_t, query_t, query_t)
        global temporal_attn
        temporal_attn=res_temporal_attn
        res_temporal = res_temporal.permute(1, 0, 2)
        # res_temporal = self.attn(query_t).permute(1, 0, 2)
        res_temporal = self.dropout_layer(
            self.proj_drop(res_temporal.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)

        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        res_temporal = res_temporal.reshape(b, p * t, m)

        # ret_value [batch_size, num_patches * num_frames + 1, embed_dims]
        new_query_t = identity + res_temporal
        new_query = torch.cat((init_cls_token, new_query_t), 1)
        return new_query




@MODELS.register_module()
class DividedSpatialAttentionWithNorm(BaseModule):
    """Spatial Attention in Divided Space Time Attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Defaults to 0..
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
            Defaults to `dict(type='DropPath', drop_prob=0.1)`.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
        init_cfg (dict | None): The Config for initialization. Defaults to
            None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

        self.init_weights()

    def init_weights(self):
        """init DividedSpatialAttentionWithNorm by default."""
        pass

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        """Defines the computation performed at every call."""
        assert residual is None, (
            'Always adding the shortcut in the forward function')


        identity = query
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]


        # query_s [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_s.size()
        p, t = pt // self.num_frames, self.num_frames

        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t,
                                                           m).unsqueeze(1)
        # query_s [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s= rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = self.norm(query_s).permute(1, 0, 2)
        res_spatial,res_spatial_attn = self.attn(query_s, query_s, query_s,average_attn_weights=True)
        global spatial_attn
        spatial_attn=res_spatial_attn
        res_spatial=res_spatial.permute(1,0,2)

        res_spatial = self.dropout_layer(
            self.proj_drop(res_spatial.contiguous()))

        # cls_token [batch_size, 1, embed_dims]
        cls_token= res_spatial[:, 0, :].reshape(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        new_query = identity + res_spatial
        return new_query
class E_MHSA_Co_Attention(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """
    def __init__(self, dim, out_dim=None, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads =num_heads
        self.head_dim=self.dim//self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        k = self.k(x)
        k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
        v = self.v(x)
        v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)


        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn=attn.permute(0,1,3,2)#dimension convert,to vertical attention
        global attn_layer
        attn_layer=attn


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)




@MODELS.register_module()
class FFNWithNorm(FFN):
    """FFN with pre normalization layer.

    FFNWithNorm is implemented to be compatible with `BaseTransformerLayer`
    when using `DividedTemporalAttentionWithNorm` and
    `DividedSpatialAttentionWithNorm`.

    FFNWithNorm has one main difference with FFN:

    - It apply one normalization layer before forwarding the input data to
        feed-forward networks.

    Args:
        embed_dims (int): Dimensions of embedding. Defaults to 256.
        feedforward_channels (int): Hidden dimension of FFNs. Defaults to 1024.
        num_fcs (int, optional): Number of fully-connected layers in FFNs.
            Defaults to 2.
        act_cfg (dict): Config for activate layers.
            Defaults to `dict(type='ReLU')`
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Defaults to 0..
        add_residual (bool, optional): Whether to add the
            residual connection. Defaults to `True`.
        dropout_layer (dict | None): The dropout_layer used when adding the
            shortcut. Defaults to None.
        init_cfg (dict): The Config for initialization. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
    """

    def __init__(self, *args, norm_cfg=dict(type='LN'), **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self, x, residual=None):
        """Defines the computation performed at every call."""
        assert residual is None, ('Cannot apply pre-norm with FFNWithNorm')
        global temporal_attn,spatial_attn
        return super().forward(self.norm(x), x), temporal_attn,spatial_attn
        # return super().forward(self.norm(x), x)



def overlay_attention(image, attention):
    # Convert attention to numpy
    attention_map = attention.detach().cpu().numpy()

    # Convert image to numpy
    image = image.cpu().detach().numpy()

    # 将 14x14 的 attention_map 扩展为 224x224（每个 token 对应 16x16 区域）
    attention_map_resized = np.kron(attention_map, np.ones((16, 16)))

    # # 确保 attention_map_resized 中只有 0 和 1
    # attention_map_resized = np.round(attention_map_resized)  # 将所有值四舍五入为 0 或 1

    # 扩展 attention_map_resized 的维度以匹配图像的通道数
    attention_map_resized = np.repeat(attention_map_resized[None,...], 3, axis=0)  # 变为 (3,224, 224)

    # # Apply attention map to image
    # 创建遮挡的副本，设置为一个稍微变暗的版本
    mask = np.ones_like(image) * 0.08  # 使用 0.2 的亮度作为遮挡

    # 对应 attention_map 为 1 的地方保留原始图像, 为 0 的地方应用遮挡
    overlay_image = np.where(attention_map_resized>=0.5, image, image * mask)


    return overlay_image.transpose(1,2,0)
    # return overlay_image



from PIL import Image
from torchvision.transforms import ToTensor

def visualize_attention(writer, images, attention_map, mask_idx,tag, nrow=8):
    num_channels,num_images,height, width =images.shape
    attention_map = attention_map.reshape((num_images, 14, 14)).cpu()  # Exclude the cls_token
    # for j in range(num_images):
    #     image = images[:, j, :, :]
    #     img_grid = vutils.make_grid(torch.tensor(image.cpu().detach().numpy().transpose(1,2,0)), nrow=nrow, normalize=True, scale_each=True)
    #     writer.add_image(f'{tag}/oringinal_img_{j}', img_grid)

    for j in range(num_images):
        image=images[:,j,:,:]
        attention = attention_map[j,:,:]
        for x1 in range(14):
            for y1 in range(14):
                if x1<=1 or x1>=12 or y1<=1 or y1>=12:
                    attention[x1][y1]=False
        attention = torch.zeros((14, 14), dtype=torch.bool)

        attention_overlay = overlay_attention(image, attention)
        img_grid = vutils.make_grid(torch.tensor(attention_overlay), nrow=8, normalize=True, scale_each=False)
        writer.add_image(f'{tag}/attention_attn_{j}', img_grid)
        # if j==0:
        #     vis_attn=torch.tensor(attention_overlay).unsqueeze(0)
        # else:
        #     vis_attn = torch.cat((vis_attn,torch.tensor(attention_overlay).unsqueeze(0)),dim=0)

    # img_grid = vutils.make_grid(vis_attn, nrow=nrow, normalize=True, scale_each=False)
    # writer.add_image(f'{tag}/attention_attn_{j}', img_grid)





from torch.nn.functional import interpolate
def visualize_attention_heatmap(writer, images, attention_scores,non_topk_indces, tag, nrow=8):
    num_channels,num_images,height, width =images.shape
    attention_scores_ = attention_scores.reshape((num_images, 14, 14)).cpu()  # Exclude the cls_token
    # 使用双线性插值将注意力图从 (14, 14) 转换到 (224, 224)
    attention_scores_resized = interpolate(attention_scores_.unsqueeze(1).to(torch.float), size=(224, 224),
                                           mode='bilinear',
                                           align_corners=False)
    for j in range(num_images):
        image = images[:, j, :, :]
        img_grid = vutils.make_grid(torch.tensor(image.cpu().detach().numpy().transpose(1,2,0)), nrow=nrow, normalize=True, scale_each=True)
        writer.add_image(f'{tag}/oringinal_img_{j}', img_grid)
    for j in range(num_images):
        image=images[:,j,:,:].permute(1,2,0).cpu().detach().numpy()
        # 3. 将 14x14 的注意力图插值到 224x224

        # 获取对应的注意力热力图，并规范化到 0-255
        attention_map = attention_scores_resized[j].squeeze().cpu().detach().numpy()
        attention_map= cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # attention_map = np.clip(attention_map, 0, 125)  # 限制最大值为 150，避免过强的热力图


        # 使用颜色映射生成热力图
        heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

        token_size=16
        for idx,val in enumerate(non_topk_indces[j]):
            # 找到对应的行列位置
            row, col = np.unravel_index(idx, (14, 14))  # 转换为 (row, col) 坐标
            # 计算对应的 16x16 区域的位置
            start_x, end_x = row * token_size, (row + 1) * token_size
            start_y, end_y = col * token_size, (col + 1) * token_size
            heatmap=heatmap.astype(np.float32)
            if val==False or row>=12 or row<=1 or col>=12 or col<=1:
                # 将该区域设置为零
                heatmap[start_x:end_x, start_y:end_y,:] *=0.2
            # else:
            #     heatmap[start_x:end_x, start_y:end_y,:] *=2


        # 将热力图与原图叠加
        superimposed_img = cv2.addWeighted(image, 0.995, heatmap.astype(np.float32), 0.005, 0)
        # img_grid = vutils.make_grid(torch.tensor(heatmap.astype(np.float32)), nrow=nrow, normalize=True, scale_each=True)
        img_grid = vutils.make_grid(torch.tensor(superimposed_img), nrow=nrow, normalize=True, scale_each=True)
        writer.add_image(f'{tag}/attention_attn_{j}', img_grid)