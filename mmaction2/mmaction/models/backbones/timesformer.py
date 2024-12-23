# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine import ConfigDict
from mmengine.logging import MMLogger
from mmengine.model.weight_init import kaiming_init, trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict
from torch.nn.modules.utils import _pair

from mmaction.registry import MODELS
# from mmengine.registry import MODELS
from mmaction.models.KTP import KTPBlock

from mmengine.visualization import TensorboardVisBackend
writer2 = TensorboardVisBackend(save_dir='/home/gaocq/code/mmaction2/feature_v15')
caculate_pic_feature=0
caculate_pic_batch_feature=1

from mmaction.editor_back.modeling.fusion_part.Frequency import Frequency_based_Token_Selection
from mmaction.editor_back.modeling.fusion_part.SFTS import SFTS
from mmaction.editor_back.modeling.backbones.vit_pytorch import BlockMask,Block,BlockFuse
from mmcv.cnn.bricks.transformer import build_dropout

class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        in_channels (int): Channel num of input features. Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        conv_cfg (dict | None): Config dict for convolution layer. Defaults to
            `dict(type='Conv2d')`.
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels=3,
                 embed_dims=768,
                 conv_cfg=dict(type='Conv2d')):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)

        num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0])
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        # Lecun norm from ClassyVision
        kaiming_init(self.projection, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the module.
        """
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x

@MODELS.register_module()
class TimeSformer(nn.Module):
    """TimeSformer. A PyTorch impl of `Is Space-Time Attention All You Need for
    Video Understanding? <https://arxiv.org/abs/2102.05095>`_

    Args:
        num_frames (int): Number of frames in the video.
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        pretrained (str | None): Name of pretrained model. Default: None.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        num_transformer_layers (int): Number of transformer layers. Defaults to
            12.
        in_channels (int): Channel num of input features. Defaults to 3.
        dropout_ratio (float): Probability of dropout layer. Defaults to 0..
        transformer_layers (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict` | None): Config of transformerlayer in
            TransformerCoder. If it is obj:`mmcv.ConfigDict`, it would be
            repeated `num_transformer_layers` times to a
            list[obj:`mmcv.ConfigDict`]. Defaults to None.
        attention_type (str): Type of attentions in TransformerCoder. Choices
            are 'divided_space_time', 'space_only' and 'joint_space_time'.
            Defaults to 'divided_space_time'.
        norm_cfg (dict): Config for norm layers. Defaults to
            `dict(type='LN', eps=1e-6)`.
    """
    supported_attention_types = [
        'divided_space_time', 'space_only', 'joint_space_time'
    ]

    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 pretrained=None,
                 embed_dims=768,
                 num_heads=12,
                 num_transformer_layers=12,
                 in_channels=3,
                 dropout_ratio=0.,
                 transformer_layers=None,
                 attention_type='divided_space_time',
                 norm_cfg=dict(type='LN', eps=1e-6),
                 keep_rates=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert attention_type in self.supported_attention_types, (
            f'Unsupported Attention Type {attention_type}!')
        assert transformer_layers is None or isinstance(
            transformer_layers, (dict, list))

        # self.num_frames = num_frames*2
        self.num_frames = num_frames
        # self.num_frames=12
        self.pretrained = pretrained
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_ratio)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(
                torch.zeros(1, self.num_frames, embed_dims))
            self.drop_after_time = nn.Dropout(p=dropout_ratio)

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        self.keep_rates=keep_rates


        if transformer_layers is None:
            # stochastic depth decay rule
            dpr = np.linspace(0, 0.1, num_transformer_layers)

            if self.attention_type == 'divided_space_time':
                _transformerlayers_cfg = [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='DividedTemporalAttentionWithNorm',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                num_frames=self.num_frames,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]),
                                norm_cfg=dict(type='LN', eps=1e-6)),
                            dict(
                                type='DividedSpatialAttentionWithNorm',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                num_frames=self.num_frames,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]),
                                norm_cfg=dict(type='LN', eps=1e-6)),
                            # dict(
                            #     type='ModalAttention',
                            #     )
                        ],
                        ffn_cfgs=dict(
                            type='FFNWithNorm',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i]),
                            norm_cfg=dict(type='LN', eps=1e-6)),
                        operation_order=('self_attn', 'self_attn','ffn'))
                    for i in range(num_transformer_layers)
                ]
            else:
                # Sapce Only & Joint Space Time
                _transformerlayers_cfg = [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                batch_first=True,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]))
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i])),
                        operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                        norm_cfg=dict(type='LN', eps=1e-6),
                        batch_first=True)
                    for i in range(num_transformer_layers)
                ]

            transformer_layers = ConfigDict(
                dict(
                    type='TransformerLayerSequence',
                    transformerlayers=_transformerlayers_cfg,
                    num_layers=num_transformer_layers))

        self.transformer_layers = build_transformer_layer_sequence(
            transformer_layers)
        self.transformer_layers_ir = build_transformer_layer_sequence(
            transformer_layers)


        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))

        # self.FREQ_INDEX = Frequency_based_Token_Selection(keep=10,
        #                                                   stride=16)
        self.SFTS = SFTS()
        self.FUSE_block = BlockMask(num_class=12, dim=768, num_heads=8, mlp_ratio=4.,
                                    qkv_bias=False, momentum=0.8)

        self.RGB_REDUCE = nn.Linear(2 * self.embed_dims, self.embed_dims)
        self.RGB_REDUCE.apply(weights_init_kaiming)
        self.NIR_REDUCE = nn.Linear(2 * self.embed_dims, self.embed_dims)
        self.NIR_REDUCE.apply(weights_init_kaiming)

        self.merged_attn=Merged_Attention(dim=self.embed_dims)
        dropout_layer = dict(type='DropPath', drop_prob=0.1)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.proj_drop = nn.Dropout(0.)

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, 0.,batch_first=True,
                                          **kwargs)


        self.norm_s1 =build_norm_layer(norm_cfg, self.embed_dims)[1]
        # self.mhsa_s = MultiheadAttention(self.model.model.embed_dim, args.num_heads)
        self.mhsa_s = nn.ModuleList(
            [MultiheadAttention(768, 8) for _ in range(12)])
        self.norm_s2 =build_norm_layer(norm_cfg, self.embed_dims)[1]
        self.mlp_s = Mlp(768)

        self.norm_t1 =build_norm_layer(norm_cfg, self.embed_dims)[1]
        # self.mhsa_t = MultiheadAttention(self.model.model.embed_dim, args.num_heads)
        self.mhsa_t = nn.ModuleList(
            [MultiheadAttention(768, 8) for _ in range(6)])
        self.norm_t2 =build_norm_layer(norm_cfg, self.embed_dims)[1]
        self.mlp_t = Mlp(768)

        self.global_cls = nn.Linear(self.embed_dims, 12)
        self.spatial_cls = nn.Linear(self.embed_dims, 12)
        self.temporal_cls = nn.Linear(self.embed_dims, 12)
        self.softmax=nn.Softmax(dim=1)

        self.teacher_head = nn.Linear(self.embed_dims,12)
        self.student_head = nn.Linear(self.embed_dims, 12)

        self.RGB_REDUCE = nn.Linear(2 * 768, 768)


    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            state_dict = _load_checkpoint(self.pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            if self.attention_type == 'divided_space_time':
                # modify the key names of norm layers
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'norms' in old_key:
                        new_key = old_key.replace('norms.0',
                                                  'attentions.0.norm')
                        new_key = new_key.replace('norms.1', 'ffns.0.norm')
                        state_dict[new_key] = state_dict.pop(old_key)

                # copy the parameters of space attention to time attention
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'attentions.0' in old_key:
                        new_key = old_key.replace('attentions.0',
                                                  'attentions.1')
                        state_dict[new_key] = state_dict[old_key].clone()

            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        """Defines the computation performed at every call."""
        # 将 transformer_layers 的权重复制到 transformer_layers_ir 中
        # self.transformer_layers_ir.load_state_dict(self.transformer_layers.state_dict())

        identity_x = x.clone()
        batches = x.shape[0]
        x = self.patch_embed(x)


        # x [batch_size * num_frames, num_patches + 1, embed_dims]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # Add Time Embedding
        if self.attention_type != 'space_only':
            # x [batch_size, num_patches * num_frames + 1, embed_dims]
            cls_tokens = x[:batches, 0, :].unsqueeze(1)
            x = rearrange(x[:, 1:, :], '(b t) p m -> (b p) t m', b=batches)
            x = x + self.time_embed
            x = self.drop_after_time(x)
            x = rearrange(x, '(b p) t m -> b (p t) m', b=batches)
            x = torch.cat((cls_tokens, x), dim=1)


        x_vis, temporal_attn_vis,spatial_attn_vis = self.transformer_layers(x[:1,:,:], None, None)
        x_ir, temporal_attn_ir, spatial_attn_ir = self.transformer_layers(x[1:,:,:], None, None)

        cls=torch.cat((x_vis[:,0],x_ir[:,0]),dim=0)
        cls=self.norm(cls)

        cls = torch.mean(cls, dim=0, keepdim=True)

        #############0\prepare stage####################
        x_cls = torch.mean(torch.cat((x_vis[:, :1, :], x_ir[:, :1, :]), dim=0), dim=0, keepdim=True)
        x_backbone = torch.cat((x_vis[:, 1:, :], x_ir[:, 1:, :]), dim=1)
        x = torch.cat((x_cls, x_backbone), dim=1)

        #############1\temporal-spatial select module####################
        for i in range(len(spatial_attn_vis)):
            if i == 0:
                last_map_vis = spatial_attn_vis[i]
                temporal_map_vis = temporal_attn_vis[i]
                last_map_ir = spatial_attn_ir[i]
                temporal_map_ir = temporal_attn_ir[i]
            else:
                last_map_vis = torch.matmul(spatial_attn_vis[i], last_map_vis)
                temporal_map_vis = torch.matmul(temporal_attn_vis[i], temporal_map_vis)
                last_map_ir = torch.matmul(spatial_attn_ir[i], last_map_ir)
                temporal_map_ir = torch.matmul(temporal_attn_ir[i], temporal_map_ir)

        mean_attn_weights_vis = temporal_map_vis.mean(dim=0)
        temporal_scores_vis = mean_attn_weights_vis.sum(dim=0)


        mean_attn_weights_ir = temporal_map_ir.mean(dim=0)
        temporal_scores_ir = mean_attn_weights_ir.sum(dim=0)

        temporal_scores=torch.cat((temporal_scores_vis, temporal_scores_ir), dim=0)
        sorted_indices = torch.argsort(temporal_scores, descending=True)
        temporal_mask = torch.zeros(16)
        temporal_mask[sorted_indices[:4]] = 2
        temporal_mask[sorted_indices[4:8]]=1

        last_map=torch.cat((last_map_vis,last_map_ir),dim=0)

        feat, mask = self.SFTS(Vis_feat=x,
                          Vis_attn=last_map,
                          mask_fre=temporal_mask,
                          identity_x=identity_x)

        #############2  adpative fuse module####################
        # feat=x
        # mask=torch.ones((1, 3136, 1), dtype=torch.bool).cuda()
        feat = self.FUSE_block(feat, mask)

        feat_s=torch.mean(feat[:,0],dim=0,keepdim=True)



        #############3  self-distillation module####################
        # feat_s=self.norm(feat)[:,0]
        mask_temp = mask.squeeze(0).squeeze(1)
        cls_kl_loss = F.kl_div(
            F.log_softmax(self.teacher_head(feat_s), dim=-1),
            F.log_softmax(self.teacher_head(cls), dim=-1),
            reduction='batchmean',
            log_target=True
        )

        token_teacher = x[:, 1:, :][:, mask_temp, :]
        token_student = feat[:, 1:, :][:, mask_temp, :]
        #there use num_mask_tokens to rescale
        # token_kl_loss = (torch.pow(token_teacher - token_student, 2).mean())/(sum(mask_temp==True)*(self.num_frames))
        token_kl_loss = (torch.pow(token_teacher - token_student, 2).mean()) / (
                    sum(mask_temp == True)*4)
        l_r=0.1
        distill_loss = l_r* token_kl_loss+ (1-l_r) * cls_kl_loss

        if self.training:
            return [feat_s, distill_loss]
        else:
            return feat_s


        # if self.training:
        #     return [feat_s]
        # else:
        #     return feat_s





def calculate_entropy(image):
    """计算图像的熵值"""
    # 计算图像的直方图
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # 归一化直方图
    # 计算熵值
    entropy = -np.sum([p * np.log2(p) for p in hist if p != 0])
    return entropy

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Merged_Attention(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """
    def __init__(self, dim, out_dim=None, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0.,):
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


    def forward(self, x_vis,x_ir):
        B, N, C = x_vis.shape

        q_vis = self.q(x_vis)
        q_vis = q_vis.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        k_vis = self.k(x_vis)
        k_vis = k_vis.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
        v_vis = self.v(x_vis)
        v_vis = v_vis.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        q_ir = self.q(x_ir)
        q_ir = q_ir.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        k_ir= self.k(x_ir)
        k_ir = k_ir.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
        v_ir = self.v(x_ir)
        v_ir = v_ir.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)


        attn_vis = (q_vis @ k_ir) * self.scale
        attn_vis = attn_vis.softmax(dim=-1)
        attn_vis = self.attn_drop(attn_vis)
        x_vis = (attn_vis @ v_ir).transpose(1, 2).reshape(B, N, C)
        x_vis = self.proj(x_vis)
        x_vis = self.proj_drop(x_vis)

        attn_ir = (q_ir @ k_vis) * self.scale
        attn_ir = attn_ir.softmax(dim=-1)
        attn_ir = self.attn_drop(attn_ir)
        x_ir = (attn_ir @ v_vis).transpose(1, 2).reshape(B, N, C)
        x_ir = self.proj(x_ir)
        x_ir= self.proj_drop(x_ir)

        return x_vis,x_ir


def temporal_attention_mask(attn_tensor, preserved_token=30):
    # 获取张量的维度信息
    num_frames, num_tokens, token_dim = attn_tensor.size()

    diff_tensor=attn_tensor[:1]
    # # 计算差分矩阵
    diff_tensor=torch.abs(torch.cat((diff_tensor,attn_tensor[1:] - attn_tensor[:-1]),dim=0)).sum(dim=-1)

    _, topk_indices = torch.topk(diff_tensor, preserved_token, dim=1)
    topk_indices = torch.sort(topk_indices, dim=1).values
    selected_tokens_mask = torch.zeros((num_frames, num_tokens), dtype=torch.bool).cuda()
    selected_tokens_mask.scatter_(1, topk_indices, 1)

    return selected_tokens_mask

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
import torch.nn.functional as F
import cv2
import torchvision.utils as vutils
def visualize_attention_on_image(features, original_images,writer,tag):
    B, H, W, D = features.shape  # 8, 14, 14, 768
    C, N, H_img, W_img = original_images.shape  # 3, 8, 224, 224

    # 平均特征维度
    avg_features = features.mean(dim=-1).detach().cpu().numpy()  # 8, 14, 14

    # 插值到原图尺寸
    upsampled_features = F.interpolate(torch.tensor(avg_features).unsqueeze(1), size=(H_img, W_img),
                                       mode='bilinear').squeeze(1).numpy()  # 8, 224, 224

    for i in range(B):
        original_image = original_images[:, i, :, :].permute(1, 2, 0).cpu().numpy()  # 224, 224, 3
        attention_map = upsampled_features[i]  # 224, 224

        # 归一化注意力图
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # 将注意力图应用颜色映射
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # 叠加注意力图在原图上
        overlay = heatmap + np.float32(original_image)
        overlay = overlay / np.max(overlay)

        img_grid = vutils.make_grid(torch.tensor(overlay), nrow=8, normalize=True, scale_each=True)
        writer.add_image(f'{tag}/attention_attn_{i}', img_grid)

class MultiheadAttention(nn.Module):
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
        attn_map = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_map)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x, attn_map


def kl_div(p, q) :
    kl = p * torch.log((p + 0.00001) / q)
    kl = torch.sum(torch.sum(kl, dim=1), dim=0) / p.size(0)
    return kl

def js_div(p, q):
    return kl_div(p, (p + q) / 2) +kl_div(q, (p + q) / 2) / 2

