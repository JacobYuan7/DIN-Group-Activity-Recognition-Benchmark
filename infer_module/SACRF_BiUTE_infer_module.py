import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from torchvision import models
from thop import profile, clever_format
from utils import print_log,MAC2FLOP
import math
from fvcore.nn import flop_count, parameter_count

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

        :param feature: B, T, N, 1024
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



class Feed_forward(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, dropout):
        super(Feed_forward, self).__init__()
        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(latent_dim, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, feature):
        feature = self.fc1(feature)
        feature = self.relu(feature)
        feature = self.dropout(feature)
        feature = self.fc2(feature)
        return feature


class Selfatt(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim):
        super(Selfatt, self).__init__()
        self.in_dim = in_dim
        self.theta = nn.Linear(in_dim, latent_dim, bias = False)
        self.phi = nn.Linear(in_dim, latent_dim, bias = False)
        self.fun_g = nn.Linear(in_dim, out_dim, bias = False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        '''
        :param x: [B, N, NFB]
        :return:
        '''
        theta_x = self.theta(x)
        phi_x = self.phi(x).transpose(1,2)
        att = torch.bmm(theta_x, phi_x) / math.sqrt(self.in_dim)
        fun_g_x = self.fun_g(x)
        att_x = torch.bmm(att, fun_g_x)
        return att_x


class Spatialatt(nn.Module):
    def __init__(self, in_dim, latent_dim, num_att = 8, dropout = 0.1,cliques = []):
        super(Spatialatt, self).__init__()
        assert latent_dim == in_dim // num_att
        self.num_att = num_att
        self.attlist = nn.ModuleList()
        for i in range(num_att):
            self.attlist.append(Selfatt(in_dim, latent_dim, latent_dim))
        self.cliques = cliques

        self.W_out = nn.Linear(in_dim, in_dim, bias=False)
        self.layernorm1 = nn.LayerNorm([in_dim])
        self.dropout1 = nn.Dropout(dropout)
        self.FFN_linear1 = nn.Linear(in_dim, in_dim, bias=True)
        self.FFN_relu = nn.ReLU(inplace=True)
        self.FFN_dropout = nn.Dropout(dropout)
        self.FFN_linear2 = nn.Linear(in_dim, in_dim, bias=True)

        self.w = nn.Parameter(torch.ones(len(self.cliques)), requires_grad=True)
        self.register_parameter('w', self.w)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, features):
        '''
        :param features: [B, T, N, NFB]
        :return:
        '''
        B, T, N, _ = features.shape
        # boxes_in_flat = boxes_in_flat.reshape(-1, N, 2)
        multi_clique = []
        for i in range(len(self.cliques)):
            clique = self.cliques[i]
            features_clique = features.view((B*T*N//clique, clique, -1))
            features_clique_atts = []
            for att in self.attlist:
                features_clique_atts.append(att(features_clique))
            features_clique_atts = torch.cat(features_clique_atts, dim = -1)
            features_clique_atts = features_clique_atts.view((B, T, N, -1))
            features_clique = features.view((B, T, N, -1))
            features_clique_atts = self.W_out(features_clique_atts)
            features_clique_atts = self.dropout1(features_clique_atts)
            features_clique_atts = self.layernorm1(features_clique_atts + features_clique)
            features_clique_atts = self.FFN_linear1(features_clique_atts)
            features_clique_atts = self.FFN_relu(features_clique_atts)
            features_clique_atts = self.FFN_dropout(features_clique_atts)
            features_clique_atts = self.FFN_linear2(features_clique_atts)

            multi_clique.append(features_clique_atts)
        multi_clique = torch.stack(multi_clique, dim = -1)
        multi_clique = torch.einsum('abcde,e->abcd', multi_clique, self.w)
        return multi_clique


class Temporalatt(nn.Module):
    def __init__(self, in_dim, latent_dim, num_att = 8, dropout = 0.1):
        super(Temporalatt, self).__init__()
        assert latent_dim == in_dim//num_att
        self.num_att = num_att
        self.attlist = nn.ModuleList()
        for i in range(num_att):
            self.attlist.append(Selfatt(in_dim, latent_dim, latent_dim))
        self.W_out = nn.Linear(in_dim, in_dim, bias = False)
        self.layernorm1 = nn.LayerNorm([in_dim])
        self.dropout1 = nn.Dropout(dropout)

        self.FFN_linear1 = nn.Linear(in_dim, in_dim, bias=True)
        self.FFN_relu = nn.ReLU(inplace=True)
        self.FFN_dropout = nn.Dropout(dropout)
        self.FFN_linear2 = nn.Linear(in_dim, in_dim, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, features):
        '''
        :param features: [B, T, N, NFB]
        :return:
        '''
        B, T, N, _ = features.shape
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.view(B*N, T, -1)

        multi_att = []
        for att in self.attlist:
            multi_att.append(att(features))
        multi_ft = torch.cat(multi_att, dim = -1)
        multi_ft = self.W_out(multi_ft)
        multi_ft = self.dropout1(multi_ft)
        multi_ft = self.layernorm1(multi_ft + features)

        # FFN
        multi_ft = self.FFN_linear1(multi_ft)
        multi_ft = self.FFN_relu(multi_ft)
        multi_ft = self.FFN_dropout(multi_ft)
        multi_ft = self.FFN_linear2(multi_ft)

        multi_ft = multi_ft.view(B, N, T, -1).contiguous()
        multi_ft = multi_ft.permute(0, 2, 1, 3)
        return multi_ft


class SACRF(nn.Module):
    def __init__(self, cfg, in_dim, num_actions, num_att = 8, dropout = 0.1, cliques=[2,3,6,12]):
        super(SACRF, self).__init__()
        # self.f_u = Feed_forward(in_dim, in_dim, num_actions, dropout)
        self.f_u = nn.Linear(in_dim, num_actions)
        self.softmax = nn.Softmax(dim = -1)

        self.PE = Embfeature_PositionEmbedding(cfg = cfg, num_pos_feats = in_dim // 2)
        self.temporal_att = Temporalatt(in_dim, in_dim//num_att, num_att = num_att)
        self.spatial_att = Spatialatt(in_dim, in_dim//num_att, num_att = num_att, dropout = dropout, cliques = cliques)
        self.f_temporal_att = nn.Linear(in_dim, num_actions, bias = False)
        self.f_spatil_att = nn.Linear(in_dim, num_actions, bias = False)
        # self.f_temporal_att = Feed_forward(in_dim, in_dim, num_actions, dropout)
        # self.f_spatil_att = Feed_forward(in_dim, in_dim, num_actions, dropout)
        self.compatible_trans_s = nn.Linear(num_actions, num_actions, bias = False)
        self.compatible_trans_t = nn.Linear(num_actions, num_actions, bias = False)
        
        # Halting
        self.halt_fc = nn.Linear(in_dim, 1, bias = True)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, features, boxes_in_flat):
        '''
        :param features: [B, T, N, NFB]
        :param boxes_in_flat: # B*T*N, 4
        :return:
        '''
        B, T, N, _ = features.shape

        features = self.PE(features, boxes_in_flat)
        C_v = features
        Q_u = self.f_u(features)
        Q_hat = self.softmax(Q_u)

        halt_prob = torch.zeros((B, T, N), dtype = torch.float32, device = features.device)
        v = 0
        halt_mask = torch.zeros((B, T, N), dtype = torch.bool, device = features.device)
        running_counter = torch.zeros((B, T, N), dtype = torch.int, device=features.device)

        while (torch.sum(~halt_mask) != 0) and v <= 9:
            spatial_ft = self.spatial_att(features)
            # spatial_ft = features
            temporal_ft = self.temporal_att(features)
            spatial_ft_f = self.f_spatil_att(spatial_ft)
            temporal_ft_f = self.f_temporal_att(temporal_ft)
            Q_p = self.compatible_trans_s(spatial_ft_f) + self.compatible_trans_t(temporal_ft_f)
            Q_bar = Q_u + Q_p
            Q_hat = self.softmax(Q_bar)

            v +=1
            C_temp = spatial_ft + temporal_ft
            C_temp[halt_mask] = C_v[halt_mask]
            C_v = C_temp

            # R(t)
            halt_prob_new = halt_prob + torch.squeeze(self.sigmoid(self.halt_fc(C_v)), dim = -1)
            halt_mask = halt_mask | (halt_prob_new>=1.)
            halt_prob_new[halt_mask] = halt_prob[halt_mask]
            halt_prob = halt_prob_new

            # N(t)
            running_counter += halt_mask.int()

        R_t = torch.sum(1. - halt_prob)
        N_t = 10 - running_counter + 1
        N_t[N_t == 11] = 10
        N_t = torch.sum(N_t)
        halt_loss = R_t + N_t

        return Q_hat, C_v, halt_loss


class BiUTE(nn.Module):
    def __init__(self, in_dim, N):
        super(BiUTE, self).__init__()
        self.q = nn.Linear(in_dim, N, bias = False)

        self.in_dim = in_dim
        latent_dim = in_dim
        self.theta_before = nn.Linear(in_dim*2, latent_dim, bias=False)
        self.phi_before = nn.Linear(in_dim*2, latent_dim, bias=False)
        self.fun_g_before = nn.Linear(in_dim*2, in_dim*2, bias=False)

        self.theta_after = nn.Linear(in_dim * 2, latent_dim, bias=False)
        self.phi_after = nn.Linear(in_dim * 2, latent_dim, bias=False)
        self.fun_g_after = nn.Linear(in_dim * 2, in_dim * 2, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        :param features: [B, T, N, NFB]
        :return:
        """
        # _, T, N, _ = features.shape
        # g_weight = self.q(features)   # [B, T, N, N]
        # g = torch.einsum('abcd,abde->abce', g_weight, features) # [B, T, N, NFB]
        # g = torch.sum(g, dim = 2)  # [B, T, NFB]
        # f, _ = torch.max(features, dim = 2)
        # n = torch.cat((g, f), dim = -1)  # [B, T, 2*NFB]

        _, T, N, _ = features.shape
        g_weight = self.q(features).transpose(2, 3)  # [B, T, N, N]
        g_weight = torch.sum(g_weight, dim = 2)[:,:,None] # [B, T, 1, N]
        g = torch.einsum('abcd,abde->abce', g_weight, features).squeeze(dim = 2)  # [B, T, N, NFB]
        f, _ = torch.max(features, dim=2)
        n = torch.cat((g, f), dim=-1)  # [B, T, 2*NFB]


        biute_n = torch.zeros(n.shape, dtype = n.dtype, device = n.device)
        for i in range(T):
            # ute before
            if i>0:
                before_n = n[:, :i]
                theta_before_n = self.theta_before(n[:,i][:,None])
                phi_before_n = self.phi_before(before_n).transpose(1, 2)
                fun_before_n = self.fun_g_before(before_n)
                before_weight = torch.bmm(theta_before_n, phi_before_n) / math.sqrt(2*self.in_dim)
                before_n = torch.bmm(before_weight, fun_before_n)
                biute_n[:,i] += before_n.squeeze(dim = 1)
            # ute after
            if i < T-1:
                after_n = n[:,(i+1):]
                theta_after_n = self.theta_after(n[:,i][:,None])
                phi_after_n = self.phi_after(after_n).transpose(1, 2)
                fun_after_n = self.fun_g_after(after_n)
                after_weight = torch.bmm(theta_after_n, phi_after_n) / math.sqrt(2*self.in_dim)
                after_n = torch.bmm(after_weight, fun_after_n)
                biute_n[:, i] += after_n.squeeze(dim = 1)
        biute_n += n



        # theta_n = self.theta(n)
        # phi_n = self.phi(n).transpose(1,2)
        # fun_g_n = self.fun_g(n)
        # biute_weight = torch.bmm(theta_n, phi_n) / math.sqrt(2*self.in_dim)
        # # biute_weight = torch.einsum('abc,adc->abd', theta_n, phi_n)
        # biute_weight = biute_weight * (1. - torch.eye(T, device = biute_weight.device)) # make the diagonal weight to be zeros
        # # for i in range(biute_weight.shape[1]):
        # #     biute_weight[:,i,i] = 0.
        # biute_n = torch.bmm(biute_weight, fun_g_n) + n
        return n # biute_n    # [B, T, 2*NFB]


