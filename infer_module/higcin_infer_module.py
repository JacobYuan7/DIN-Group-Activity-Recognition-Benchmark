import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from torchvision import models
from thop import profile, clever_format
from utils import print_log

class CrossInferBlock(nn.Module):
    def __init__(self, in_dim, Temporal, Spatial):
        super(CrossInferBlock, self).__init__()
        latent_dim = in_dim//2
        field = Temporal + Spatial

        self.theta = nn.Linear(in_dim, latent_dim, bias = False)
        self.phi = nn.Linear(in_dim, latent_dim, bias = False)
        self.fun_g = nn.Linear(in_dim, latent_dim, bias = False)
        self.W = nn.Linear(latent_dim, in_dim, bias = False)
        self.bn = nn.BatchNorm2d(in_dim)
        # self.embedding = nn.Linear(in_dim, latent_dim, bias = True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        :param x: shape [B, T, N, NFB]
        :return:
        '''
        B, T, N, NFB = x.shape
        newx = x.clone()
        for i in range(T):
            for j in range(N):
                x_ij = x[:, i, j, :] # [B, NFB]
                embed_x_ij = self.theta(x_ij).unsqueeze(dim = 2) # [B, NFB//2, 1]

                # Spatial
                spatio_x = x[:,i] # [B, N, NFB]
                g_spatial = self.fun_g(spatio_x)
                phi_spatio_x = self.phi(spatio_x) # [B, N, NFB//2]
                    # Original paper does not use softmax, thus we stick to it
                sweight = torch.bmm(phi_spatio_x, embed_x_ij).squeeze(dim = 2) # [B,N]
                n = len(sweight[0,:])
                spatio_info = torch.einsum('ij,ijk->ik', sweight/n, g_spatial)

                # Temporal
                temporal_x = x[:,:,j]
                g_temporal = self.fun_g(temporal_x)
                embed_temporal_x = self.phi(temporal_x)
                    # Original paper does not use softmax, thus we stick to it
                tweight = torch.bmm(embed_temporal_x, embed_x_ij).squeeze(dim = 2)
                n = len(tweight[0,:])
                temporal_info = torch.einsum('ij,ijk->ik', tweight/n, g_temporal)

                ST_info = (spatio_info + temporal_info)/(T+N)
                res_ST_info = self.W(ST_info) + x_ij
                newx[:,i,j,:] = res_ST_info

        newx = newx.permute(0, 3, 1, 2)
        newx = self.bn(newx)
        newx = newx.permute(0, 2, 3, 1)

        return newx

def MAC2FLOP(macs, params, module_name = ''):
    macs, params = clever_format([macs, params], "%.3f")
    print('{} MACs: {}  #Params: {}'.format(module_name, macs, params))
    if 'M' in macs:
        flops = float(macs.replace('M', '')) * 2
        flops = str(flops/1000) + 'G'
    elif 'G' in macs:
        flops = str(float(macs.replace('G', '')) * 2) + 'G'
    print('{} GFLOPs: {}  #Params: {}'.format(module_name, flops, params))




