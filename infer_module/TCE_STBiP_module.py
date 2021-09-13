import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils import calc_pairwise_distance_3d
# from hrnet.init_hrnet import cls_hrnet_w32, pose_hrnet_w32
from config import Config

#################       Bilinear Pooling Reasoning Module        ###################

class STBilinearMessagingPassing(nn.Module):
    def __init__(self, emb_fea_num, message_fea_num, T=3):
        super(STBilinearMessagingPassing, self).__init__()
        print('The emb_fea_num of Bilinear is ' + str(emb_fea_num))
        print('The message_fea_num of Bilinear is ' + str(message_fea_num))
        self.T = T
        '''
        self.U = nn.Linear(emb_fea_num, message_fea_num, bias=True)
        self.V = nn.Linear(emb_fea_num, message_fea_num, bias=True)
        self.w_a = nn.Parameter(torch.FloatTensor(1, message_fea_num), requires_grad=True)
        '''
        self.U = nn.Linear(emb_fea_num, emb_fea_num, bias=True)
        self.V = nn.Linear(emb_fea_num, emb_fea_num, bias=True)
        self.w_a = nn.Parameter(torch.FloatTensor(1, emb_fea_num), requires_grad=True)
        # jianshu.com/p/d8b77cc02410
        nn.init.kaiming_normal_(self.w_a)

        self.W_e2 = nn.Linear(emb_fea_num, message_fea_num, bias=False)
        self.W_e1 = nn.Linear(message_fea_num, emb_fea_num, bias=False)
        self.layernorm = nn.LayerNorm(message_fea_num)
        self.non_linear = nn.ReLU(inplace=True)
        #nn.init.kaiming_normal_(self.W_e1.weight)
        #nn.init.kaiming_normal_(self.W_e2.weight)
        self.R_mat = None


    def forward(self, feature, mask):
        '''
        :param feature: shape:[B*T, N, NFB]
        :param mask: [B*T, N, N]
        :return: [B*T, N, NFB]
        '''
        T = self.T
        B = feature.shape[0]//T
        #BT = feature.shape[0]
        N = feature.shape[1]
        feature = feature.reshape(B, T*N, -1)
        # feature = feature.reshape(BT*N, -1)
        feature_U = self.U(feature)  # [B, T*N, NFM]
        # feature_U = feature_U.reshape(BT, N, -1)
        feature_V = self.V(feature)  # [B, T*N, NFM]
        # feature_V = feature_V.reshape(BT, N, -1)

        feature_U = feature_U * self.w_a  # [B, T*N, NFM]
        UV = torch.matmul(feature_U, feature_V.transpose(1, 2))  # [B, T*N, T*N]
        UV[mask] = -float('inf')
        # print("UV shape:"+str(UV.shape))
        matrix_e = F.softmax(UV, dim=2)  # [B, T*N, T*N] softmax by row!!!!!
        self.R_mat = matrix_e

        feature_W_e2 = self.W_e2(feature)
        feature_e = torch.matmul(matrix_e, feature_W_e2)  # [B, T*N, NFM]
        feature_e_nl = self.layernorm(feature_e)  # [B, T*N, NFM]
        feature_e_nl_nonl = self.non_linear(feature_e_nl)  # [B, T*N, NFM]
        feature_out = self.W_e1(feature_e_nl_nonl)

        feature_out = feature_out.reshape(B*T, N, -1)
        return feature_out

class multiheadSTBilinearMessagingPassing(nn.Module):
    def __init__(self, emb_fea_num, message_fea_num, num_heads, T=3):
        super(multiheadSTBilinearMessagingPassing, self).__init__()
        self.bilinear_list = nn.ModuleList([STBilinearMessagingPassing(emb_fea_num, message_fea_num, T=T) for i in range(num_heads)])
        self.num_heads = num_heads
        self.vis_R_mat = torch.empty((0, 36, 36), dtype=torch.float32)

    def forward(self, feature, mask, fusion_method, shortcut_connection = False):
        if fusion_method == 'sum':
            feature_out = self.bilinear_list[0](feature, mask)
            #self.vis_R_mat = torch.cat((self.bilinear_list[0].R_mat.cpu(), self.vis_R_mat), dim = 0)
            for i in range(self.num_heads - 1):
                feature_out+=self.bilinear_list[i+1](feature, mask)
                #self.vis_R_mat = torch.cat((self.bilinear_list[i+1].R_mat.cpu(), self.vis_R_mat),
                #                           dim = 0)
        elif fusion_method == 'cat':
            feature_out = []
            for i in range(self.num_heads):
                feature_out.append(self.bilinear_list[i](feature, mask))
            feature_out = torch.cat(feature_out, dim = 2)

        #print(self.vis_R_mat.shape[0])
        #if self.vis_R_mat.shape[0] == 20*3*8:
        #    save_R_mat = self.vis_R_mat.numpy().reshape(20*3*8, 36*36)
        #    np.savetxt('vis/R_mat/R_mat.txt', save_R_mat)

        if fusion_method == 'sum':
            if shortcut_connection == True:
                return feature + feature_out
            elif shortcut_connection == False:
                return feature_out
        elif fusion_method == 'cat':
            return torch.cat((feature_out, feature), dim=2)



def generate_spatial_mask(boxes_positions, threshold, BT, N, OH):
    """
    :param loc:  B*T*N, 4 #Center point of every box
    :param threshold: float, e.g. 0.3, 0.2
    :return:
    """
    boxes_positions_cl = boxes_positions.clone()
    boxes_positions_cl[:, 0] = (boxes_positions_cl[:, 0] + boxes_positions_cl[:, 2]) / 2
    boxes_positions_cl[:, 1] = (boxes_positions_cl[:, 1] + boxes_positions_cl[:, 3]) / 2
    boxes_positions_cl = boxes_positions_cl[:, :2].reshape(BT, N, 2)  # B*T, N, 2

    boxes_distances = calc_pairwise_distance_3d(boxes_positions_cl, boxes_positions_cl)  # B*T, N, N
    position_mask = (boxes_distances > (threshold * OH))

    return position_mask



#################       Context Encoding Module        ###################
#################       Context Encoding Module        ###################
#################       Context Encoding Module        ###################

class ContextEncodingTransformer(nn.Module):
    def __init__(self, num_features_context, D, K, N, layer_id, num_heads_per_layer, context_dropout_ratio = 0.1):
        super(ContextEncodingTransformer, self).__init__()
        self.num_features_context = num_features_context
        if layer_id == 1:

            self.downsample1 = nn.Conv2d(D, num_features_context, kernel_size = 1, stride = 1)
            self.downsample2 = nn.Conv2d(768, num_features_context, kernel_size = 1, stride=1)
            '''nn.init.kaiming_normal_(self.downsample1.weight)
            nn.init.kaiming_normal_(self.downsample2.weight)
            self.downsample = nn.Conv2d(D, num_features_context, kernel_size=1, stride=1)'''
            #nn.init.kaiming_normal_(self.downsample.weight)
            self.emb_roi = nn.Linear(num_features_context * K * K, num_features_context, bias=True)
        elif layer_id > 1:
            self.downsample = nn.Conv2d(768, num_features_context, kernel_size=1, stride=1)
            self.emb_roi = nn.Linear(num_features_context * num_heads_per_layer, num_features_context, bias=True)
            nn.init.kaiming_normal_(self.downsample.weight)
        self.N = N
        self.K = K
        self.dropout = nn.Dropout(context_dropout_ratio)
        self.layernorm1 = nn.LayerNorm(num_features_context)
        self.FFN = nn.Sequential(
            nn.Linear(num_features_context,num_features_context, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(context_dropout_ratio),
            nn.Linear(num_features_context,num_features_context, bias = True)
        )
        self.layernorm2 = nn.LayerNorm(num_features_context)



    def forward(self, roi_feature, image_feature, layer_id = -1):
        """

        :param roi_feature:   # B*T*N, D, K, K,
        :param image_feature: # B*T, D, OH, OW
        :return:
        """
        NFC = self.num_features_context
        BT, _,OH,OW = image_feature.shape
        K = self.K #roi_feature.shape[3]
        N = self.N #roi_feature.shape[0]//BT
        assert N==12
        assert layer_id>=1
        if layer_id == 1:
            roi_feature = self.downsample1(roi_feature)
            image_feature = self.downsample2(image_feature)
            roi_feature = roi_feature.reshape(-1, NFC*K*K)
            emb_roi_feature = self.emb_roi(roi_feature) # B*T*N, D
        elif layer_id > 1:
            emb_roi_feature = self.emb_roi(roi_feature)
            image_feature = self.downsample(image_feature)
        emb_roi_feature = emb_roi_feature.reshape(BT, N, 1, 1, NFC) # B*T, N, 1, 1, D
        image_feature = image_feature.reshape(BT, 1, NFC, OH, OW) # B*T, 1, D, OH, OW
        image_feature = image_feature.transpose(2,3) # B*T, 1, OH, D, OW

        a = torch.matmul(emb_roi_feature, image_feature) # B*T, N, OH, 1, OW
        a = a.reshape(BT, N, -1) # B*T, N, OH*OW
        A = F.softmax(a, dim=2)  # B*T, N, OH*OW
        image_feature = image_feature.transpose(3,4).reshape(BT, OH*OW, NFC)

        context_encoding_roi = self.dropout(torch.matmul(A, image_feature).reshape(BT*N, NFC))
        emb_roi_feature = emb_roi_feature.reshape(BT*N, NFC)
        context_encoding_roi = self.layernorm1(context_encoding_roi + emb_roi_feature)
        context_encoding_roi = context_encoding_roi + self.FFN(context_encoding_roi)
        context_encoding_roi = self.layernorm2(context_encoding_roi)
        return context_encoding_roi


class MultiHeadLayerContextEncoding(nn.Module):
    def __init__(self, num_heads_per_layer, num_layers, num_features_context, D, K, N, context_dropout_ratio=0.1):
        super(MultiHeadLayerContextEncoding, self).__init__()
        self.CET = nn.ModuleList()
        for i in range(num_layers):
            for j in range(num_heads_per_layer):
                self.CET.append(ContextEncodingTransformer(num_features_context, D, K, N, i+1, num_heads_per_layer, context_dropout_ratio))
        self.num_layers = num_layers
        self.num_heads_per_layer = num_heads_per_layer

    def forward(self, roi_feature, image_feature):
        """
        :param roi_feature:   # B*T*N, D, K, K,
        :param image_feature: # B*T, D, OH, OW
        :return:
        """
        for i in range(self.num_layers):
            MHL_context_encoding_roi= []
            for j in range(self.num_heads_per_layer):
                MHL_context_encoding_roi.append(self.CET[i*self.num_heads_per_layer + j](roi_feature, image_feature, i+1))
            roi_feature = torch.cat(MHL_context_encoding_roi, dim=1)


        return roi_feature


class EmbfeatureContextEncodingTransformer(nn.Module):
    def __init__(self, num_features_context, NFB, K, N, layer_id, num_heads_per_layer, context_dropout_ratio = 0.1):
        super(EmbfeatureContextEncodingTransformer, self).__init__()
        self.num_features_context = num_features_context
        if layer_id == 1:
            self.downsample2 = nn.Conv2d(512, num_features_context, kernel_size = 1, stride=1)
            '''nn.init.kaiming_normal_(self.downsample1.weight)
            nn.init.kaiming_normal_(self.downsample2.weight)
            self.downsample = nn.Conv2d(D, num_features_context, kernel_size=1, stride=1)'''
            self.emb_roi = nn.Linear(NFB, num_features_context, bias=True)
        elif layer_id > 1:
            self.downsample = nn.Conv2d(512, num_features_context, kernel_size=1, stride=1)
            self.emb_roi = nn.Linear(num_features_context * num_heads_per_layer, num_features_context, bias=True)
            nn.init.kaiming_normal_(self.downsample.weight)
        self.N = N
        self.K = K
        self.dropout = nn.Dropout(context_dropout_ratio)
        self.layernorm1 = nn.LayerNorm(num_features_context)
        self.FFN = nn.Sequential(
            nn.Linear(num_features_context,num_features_context, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(context_dropout_ratio),
            nn.Linear(num_features_context,num_features_context, bias = True)
        )
        self.layernorm2 = nn.LayerNorm(num_features_context)
        self.att_map = None


    def forward(self, roi_feature, image_feature, layer_id = -1):
        """

        :param roi_feature:   # B*T*N, NFB
        :param image_feature: # B*T, D, OH, OW
        :return:
        """
        NFC = self.num_features_context
        BT, _,OH,OW = image_feature.shape
        K = self.K #roi_feature.shape[3]
        N = self.N #roi_feature.shape[0]//BT
        assert N==12
        assert layer_id>=1
        if layer_id == 1:
            image_feature = self.downsample2(image_feature)
            emb_roi_feature = self.emb_roi(roi_feature) # B*T*N, D
        elif layer_id > 1:
            emb_roi_feature = self.emb_roi(roi_feature)
            image_feature = self.downsample(image_feature)
        emb_roi_feature = emb_roi_feature.reshape(BT, N, 1, 1, NFC) # B*T, N, 1, 1, D
        image_feature = image_feature.reshape(BT, 1, NFC, OH, OW) # B*T, 1, D, OH, OW
        image_feature = image_feature.transpose(2,3) # B*T, 1, OH, D, OW

        a = torch.matmul(emb_roi_feature, image_feature) # B*T, N, OH, 1, OW
        a = a.reshape(BT, N, -1) # B*T, N, OH*OW
        A = F.softmax(a, dim=2)  # B*T, N, OH*OW
        self.att_map = A
        image_feature = image_feature.transpose(3,4).reshape(BT, OH*OW, NFC)

        context_encoding_roi = self.dropout(torch.matmul(A, image_feature).reshape(BT*N, NFC))
        emb_roi_feature = emb_roi_feature.reshape(BT*N, NFC)
        context_encoding_roi = self.layernorm1(context_encoding_roi + emb_roi_feature)
        context_encoding_roi = context_encoding_roi + self.FFN(context_encoding_roi)
        context_encoding_roi = self.layernorm2(context_encoding_roi)
        return context_encoding_roi


class MultiHeadLayerEmbfeatureContextEncoding(nn.Module):
    def __init__(self, num_heads_per_layer, num_layers, num_features_context, NFB, K, N, context_dropout_ratio=0.1):
        super(MultiHeadLayerEmbfeatureContextEncoding, self).__init__()
        self.CET = nn.ModuleList()
        for i in range(num_layers):
            for j in range(num_heads_per_layer):
                self.CET.append(EmbfeatureContextEncodingTransformer(num_features_context, NFB, K, N, i+1, num_heads_per_layer, context_dropout_ratio))
        self.num_layers = num_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.vis_att_map = torch.empty((0, 12, 43 * 78), dtype = torch.float32)

    def forward(self, roi_feature, image_feature):
        """
        :param roi_feature:   # B*T*N, NFB,
        :param image_feature: # B*T, D, OH, OW
        :return:
        """
        for i in range(self.num_layers):
            MHL_context_encoding_roi= []
            for j in range(self.num_heads_per_layer):
                MHL_context_encoding_roi.append(self.CET[i*self.num_heads_per_layer + j](roi_feature, image_feature, i+1))
            roi_feature = torch.cat(MHL_context_encoding_roi, dim=1)

        return roi_feature




#################       Pose Encoding Module        ###################
#################       Pose Encoding Module        ###################
#################       Pose Encoding Module        ###################

img_name = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg',
            '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg']

class Pose2d_Encoder(nn.Module):
    def __init__(self, cfg, pose_net = 'pose_hrnet_w32'):
        super(Pose2d_Encoder, self).__init__()
        if pose_net == 'pose_hrnet_w32':
            self.encoder = pose_hrnet_w32(pretrained=True)
        self.cfg = cfg

    def forward(self, image, boxes):
        """
        :param image: # B*T, 3, H, W     # after mean and std tranform
        :param boxes: # B*T*N, 4    # w1, h1, w2, h2    #OH, OW
        :return:
        """
        BT = image.shape[0]
        N = int(boxes.shape[0] / BT)
        OH, OW = self.cfg.out_size
        H, W = self.cfg.image_size
        #assert N == 12
        ori_boxes = boxes.clone()
        ori_boxes = ori_boxes.cpu().numpy()
        ori_boxes[:,0] = np.clip(ori_boxes[:,0] / float(OW) * float(W), 0, W)
        ori_boxes[:,2] = np.clip(ori_boxes[:,2] / float(OW) * float(W), 0, W)
        ori_boxes[:,1] = np.clip(ori_boxes[:,1] / float(OH) * float(H), 0, H)
        ori_boxes[:,3] = np.clip(ori_boxes[:,3] / float(OH) * float(H), 0, H)
        ori_boxes = ori_boxes.reshape(BT, N, 4) #BT, N, 4

        roi_image = []
        for i in range(BT):
            for j in range(N):
                ij_box = (int(ori_boxes[i][j][1]), int(ori_boxes[i][j][3]), int(ori_boxes[i][j][0]), int(ori_boxes[i][j][2]))
                roi_image.append(image[i, :, ij_box[0]:ij_box[1], ij_box[2]:ij_box[3]])
                roi_image[-1] = roi_image[-1].cpu().numpy()
                roi_image[-1] = roi_image[-1].transpose(1,2,0) # 3,H,W ->H,W,3
                roi_image[-1] = cv2.resize(roi_image[-1], (192, 256))
                #cv2.imwrite(img_name[j], roi_image[-1]*255.)
                roi_image[-1] = torch.Tensor(roi_image[-1].transpose(2,0,1)) # H,W,3 ->3,H,W

        roi_image = torch.stack(roi_image, dim = 0) # B*T*N, 3, H, W
        roi_image = roi_image.cuda()
        #print(roi_image.shape) #torch.Size([72, 3, 256, 192])
        roi_pose_feature = self.encoder(roi_image)
        return roi_pose_feature







if __name__=='__main__':
    '''test SpatialMessagePassing
    s = SpatialMessagePassing(4, 4)
    t = torch.rand(1,4,4)
    mask = torch.ones((1,4,4))
    print(s(t, mask))
    print(t)'''

    '''test Pose2d_Encoder
    cfg = Config('volleyball')
    p2d = Pose2d_Encoder(cfg)'''

    '''test Context Encoding Transformer
    cet = ContextEncodingTransformer(num_features_context=128,D=256, K=5, N=12, layer_id=1,
                                     num_heads_per_layer=1, context_dropout_ratio = 0.1)
    roi_feature = torch.rand(36,256,5,5)
    image_feature = torch.rand(3, 256, 45, 80)
    context_encoding_roi = cet(roi_feature, image_feature, 1)
    print(context_encoding_roi.shape)'''


    '''test multi-layer multi-head context encoding transformer'''
    mlhcet = MultiHeadLayerContextEncoding(3, 1, num_features_context=128,  D=256, K=5, N=12, context_dropout_ratio=0.1)
    roi_feature = torch.rand(36, 256, 5, 5)
    image_feature = torch.rand(3, 256, 45, 80)
    context_encoding_roi = mlhcet(roi_feature, image_feature)
    print(context_encoding_roi.shape)

    '''test temporal message passing
    tmp =  multiheadTemporalMessage(128, 128, 3)
    t1 = torch.rand(6,12,128)
    mask = generate_temporal_mask(2, 12, 3)
    print(mask.shape)
    output = tmp(t1, mask, shortcut_connection=True)
    print(output)
    print(output.shape)'''
