import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils import calc_pairwise_distance_3d, MAC2FLOP
from thop import profile
import math
from thop import profile, clever_format
from config import Config

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensors, mask):
        x = tensors
        mask = mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # B, H, W
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # B, H, W
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # C,
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # C,

        pos_x = x_embed[:, :, :, None] / dim_t # B, H, W / C, -> B, H, W, C
        pos_y = y_embed[:, :, :, None] / dim_t # B, H, W / C, -> B, H, W, C
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # B, H, W, C/2, 2 -> B, H, W, C (in sin, cos, sin, cos order)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # B, H, W, 2C -> B, 2C, H, W
        return pos


class Embfeature_PositionEmbedding(nn.Module):
    def __init__(self, cfg, num_pos_feats=512, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.image_size = cfg.image_size # 720, 1280
        self.out_size = cfg.out_size # 45, 80
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, feature, boxes_in_flat):
        '''

        :param feature: B * T * N, 1024
        :param boxes_in_flat: B * T * N, 4
        :return:
        '''
        B, T, N, NFB = feature.shape
        feature = feature.view((B*T*N, NFB))

        assert self.num_pos_feats*2 == feature.shape[1]
        out_boxes_x = (boxes_in_flat[:,0] + boxes_in_flat[:,2]) / 2.
        out_boxes_y = (boxes_in_flat[:,1] + boxes_in_flat[:,3]) / 2.
        image_boxes_x = out_boxes_x * self.image_size[1] / self.out_size[1] # B * T * N,
        image_boxes_y = out_boxes_y * self.image_size[0] / self.out_size[0] # B * T * N,

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=feature.device)  # C,
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # C,

        pos_x = image_boxes_x[:,None] / dim_t
        pos_y = image_boxes_y[:,None] / dim_t
        pos_x = torch.stack((pos_x[:,0::2].sin(), pos_x[:,1::2].cos()), dim = 2).flatten(1)
        pos_y = torch.stack((pos_y[:,0::2].sin(), pos_y[:,1::2].cos()), dim = 2).flatten(1)

        pos_emb = torch.cat((pos_x, pos_y), dim = 1)
        assert pos_emb.shape == feature.shape
        feature_emb = pos_emb + feature

        feature_emb = feature_emb.view((B, T, N, NFB))

        return feature_emb



class Actor_Transformer(nn.Module):
    def __init__(self, in_dim, temporal_pooled_first, dropout = 0.1):
        super(Actor_Transformer, self).__init__()
        self.in_dim = in_dim
        self.temporal_pooled_first = temporal_pooled_first
        self.Q_W = nn.Linear(in_dim, in_dim, bias = False)
        self.K_W = nn.Linear(in_dim, in_dim, bias = False)
        self.V_W = nn.Linear(in_dim, in_dim, bias = False)
        self.layernorm1 = nn.LayerNorm([in_dim])
        self.dropout1 = nn.Dropout(dropout)

        self.FFN_linear1 = nn.Linear(in_dim, in_dim, bias = True)
        self.FFN_relu = nn.ReLU(inplace=True)
        self.FFN_dropout = nn.Dropout(dropout)
        self.FFN_linear2 = nn.Linear(in_dim, in_dim, bias=True)

        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm([in_dim])

    def forward(self, x):
        '''
        :param x: shape [B, T, N, NFB]
        :return:
        '''
        B, T, N, NFB = x.shape
        if self.temporal_pooled_first:
            x = torch.mean(x, dim = 1)
        else:
            x = x.view(B*T, N, NFB)

        query = self.Q_W(x)
        keys = self.K_W(x).transpose(1, 2)
        values = self.V_W(x)
        att_weight = torch.bmm(query, keys) / math.sqrt(self.in_dim)
        # att_weight = torch.einsum('bnc,bcm->bnm', query, keys) / math.sqrt(self.in_dim)
        att_weight = torch.softmax(att_weight, dim = -1)
        att_values = torch.bmm(att_weight, values)

        x = self.layernorm1(x + self.dropout1(att_values))
        FFN_x = self.FFN_linear1(x)
        FFN_x = self.FFN_relu(FFN_x)
        FFN_x = self.dropout2(FFN_x)
        FFN_x = self.FFN_linear2(FFN_x)
        x = self.layernorm2(x + self.dropout2(FFN_x))
        return x

if __name__=='__main__':
    AT = Actor_Transformer(1024, False)
    features = torch.randn((1, 10, 12, 1024))
    print(AT(features).shape)
    macs, params = profile(AT, inputs=(features,))
    MAC2FLOP(macs, params, module_name='AT')





