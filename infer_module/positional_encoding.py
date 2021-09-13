import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import cv2
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



class Context_PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, context_downscale_ratio, num_pos_feats, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.context_downscale_ratio = context_downscale_ratio
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, context):
        x = context
        mask_shape = (context.shape[0], context.shape[2], context.shape[3])
        mask = torch.ones(mask_shape , device = context.device) == 0 # All False
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) * self.context_downscale_ratio # B, H, W
        x_embed = not_mask.cumsum(2, dtype=torch.float32) * self.context_downscale_ratio # B, H, W
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

        context_pos = context + pos
        return context_pos


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

        return feature_emb






if __name__ == '__main__':
    ''' test PositionEmbeddingSine
    pe = PositionEmbeddingSine(4, 10000, False, None)
    mask = torch.ones(1,2,4) == 0
    tensors = torch.rand(1,2,2,4)
    print(pe(tensors, mask).shape)
    print(pe(tensors, mask))'''

    ''' test Embfeature_PositionEmbedding '''
    cfg = Config('HrBase_volleyball')
    #cfg = Config('InvReason_volleyball')
    EP = Embfeature_PositionEmbedding(cfg, num_pos_feats=512)
    feature = torch.randn(12, 1024)
    boxes_in_flat = torch.randn(12, 4)
    feature_emb = EP(feature, boxes_in_flat)
    print(feature_emb.shape)

    ''' test Context_PositionEmbeddingSine '''
    CP = Context_PositionEmbeddingSine(8, 128/2)
    context = torch.randn(1, 128, 45, 80)
    context_emb = CP(context)
    print(context_emb.shape)

