from backbone.backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from infer_module.dynamic_infer_module import Dynamic_Person_Inference, Hierarchical_Dynamic_Inference, Multi_Dynamic_Inference
from infer_module.pctdm_infer_module import PCTDM
from infer_module.higcin_infer_module import CrossInferBlock
from infer_module.AT_infer_module import Actor_Transformer, Embfeature_PositionEmbedding
from infer_module.ARG_infer_module import GCN_Module
from infer_module.SACRF_BiUTE_infer_module import SACRF, BiUTE
from infer_module.TCE_STBiP_module import MultiHeadLayerEmbfeatureContextEncoding
from infer_module.positional_encoding import Context_PositionEmbeddingSine
import collections


class Dynamic_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(Dynamic_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph
        
        
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained = True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained = True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained = True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K*K*D,NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])
        
        
        #self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        if not self.cfg.hierarchical_inference:
            # self.DPI = Dynamic_Person_Inference(
            #     in_dim = in_dim,
            #     person_mat_shape = (10, 12),
            #     stride = cfg.stride,
            #     kernel_size = cfg.ST_kernel_size,
            #     dynamic_sampling=cfg.dynamic_sampling,
            #     sampling_ratio = cfg.sampling_ratio, # [1,2,4]
            #     group = cfg.group,
            #     scale_factor = cfg.scale_factor,
            #     beta_factor = cfg.beta_factor,
            #     parallel_inference = cfg.parallel_inference,
            #     cfg = cfg)
            self.DPI = Multi_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape = (10, 12),
                stride = cfg.stride,
                kernel_size = cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio = cfg.sampling_ratio, # [1,2,4]
                group = cfg.group,
                scale_factor = cfg.scale_factor,
                beta_factor = cfg.beta_factor,
                parallel_inference = cfg.parallel_inference,
                num_DIM = cfg.num_DIM,
                cfg = cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg = cfg,)
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, N, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)


        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size = 1, stride = 1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities=nn.Linear(NFG, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k,v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num +=1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num)+' parameters loaded for '+prefix)


    def forward(self,batch_data):
        images_in, boxes_in = batch_data
        
        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        
        
        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K

        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features=self.nl_emb_1(boxes_features)
        boxes_features=F.relu(boxes_features, inplace = True)

        if self.cfg.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace = True)
        else:
            None

        # Dynamic graph inference
        # graph_boxes_features = self.DPI(boxes_features)
        graph_boxes_features, ft_infer_MAD = self.DPI(boxes_features)
        torch.cuda.empty_cache()


        if self.cfg.backbone == 'res18':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'vgg16':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace = True)
            boxes_states = self.dropout_global(boxes_states)


        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        
        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states,dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B*T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num
        
        # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores,dim=1).reshape(B,-1)

        return {'activities':activities_scores} # actions_scores, activities_scores


class Dynamic_TCE_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(Dynamic_TCE_volleyball, self).__init__()
        self.cfg = cfg
        num_heads_context = 4
        num_features_context = 128

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        # TCE Module Loading
        self.multilayer_head_embfeature_context_encoding = \
            MultiHeadLayerEmbfeatureContextEncoding(
                num_heads_context, 1,
                num_features_context, NFB, K, N, context_dropout_ratio=0.1)
        self.context_positionembedding1 = Context_PositionEmbeddingSine(16, 512 / 2)

        # DIN
        context_dim = in_dim + num_heads_context * num_features_context
        if not self.cfg.hierarchical_inference:
            # self.DPI = Dynamic_Person_Inference(
            #     in_dim = in_dim,
            #     person_mat_shape = (10, 12),
            #     stride = cfg.stride,
            #     kernel_size = cfg.ST_kernel_size,
            #     dynamic_sampling=cfg.dynamic_sampling,
            #     sampling_ratio = cfg.sampling_ratio, # [1,2,4]
            #     group = cfg.group,
            #     scale_factor = cfg.scale_factor,
            #     beta_factor = cfg.beta_factor,
            #     parallel_inference = cfg.parallel_inference,
            #     cfg = cfg)
            self.DPI = Multi_Dynamic_Inference(
                in_dim=context_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                num_DIM=cfg.num_DIM,
                cfg=cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim=context_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg=cfg, )
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, N, context_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size=1, stride=1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities = nn.Linear(context_dim, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        if self.cfg.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace=True)
        else:
            None

        # Context Positional Encoding
        context = outputs[-1]
        context = self.context_positionembedding1(context)
        # Embedded Feature Context Encoding
        context_states = self.multilayer_head_embfeature_context_encoding(boxes_features, context)
        context_states = context_states.reshape(B, T, N, -1)
        boxes_features = torch.cat((boxes_features, context_states), dim=3)

        # Dynamic graph inference
        # graph_boxes_features = self.DPI(boxes_features)
        graph_boxes_features, ft_infer_MAD = self.DPI(boxes_features)
        torch.cuda.empty_cache()

        if self.cfg.backbone == 'res18':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'vgg16':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace=True)
            boxes_states = self.dropout_global(boxes_states)

        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num


        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return {'activities': activities_scores}  # actions_scores, activities_scores



class PCTDM_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(PCTDM_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.pctdm = PCTDM(cfg)
        self.pctdm_nl = nn.LayerNorm([T, 2000])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        self.fc_activities = nn.Linear(2000, self.cfg.num_activities)
        self.fc_actions = nn.Linear(2000, self.cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes


        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        # PCTDM inference
        graph_boxes_features = self.pctdm(boxes_features)
        torch.cuda.empty_cache()

        boxes_states = graph_boxes_features.view((B, T, -1))
        boxes_states = self.pctdm_nl(boxes_states)
        boxes_states = F.relu(boxes_states, inplace=True)
        boxes_states = self.dropout_global(boxes_states)

        # Predict actions
        # actions_scores = self.fc_actions(boxes_states)
        # actions_scores = actions_scores.view((B, T, N, -1))
        # actions_scores = torch.mean(actions_scores, dim = 1)
        # actions_scores = actions_scores.view(B*N, -1)

        # Predict activities
        # boxes_states_pooled, _ = torch.max(boxes_states, dim=1)
        boxes_states_pooled_flat = boxes_states.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)


        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)

        # Predict activities
        # boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        # boxes_states_pooled_flat = boxes_states
        # activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num
        # activities_scores = activities_scores.reshape(B, T, -1)
        # activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return {'activities':activities_scores} # [actions_scores, activities_scores] # activities_scores


class HiGCIN_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(HiGCIN_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.person_avg_pool = nn.AvgPool2d((K**2, 1), stride = 1)
        self.BIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = K**2)
        self.PIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = N)
        self.dropout = nn.Dropout()
        self.fc_activities = nn.Linear(D, cfg.num_activities, bias = False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]


        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.view(B, T, N, D, K*K)
        boxes_features = boxes_features.permute(0, 2, 1, 4, 3).contiguous()
        boxes_features = boxes_features.view(B*N, T, K*K, D) # B*N, T, K*K, D

        # HiGCIN Inference
        boxes_features = self.BIM(boxes_features) # B*N, T, K*K, D
        boxes_features = self.person_avg_pool(boxes_features) # B*N, T, D
        boxes_features = boxes_features.view(B, N, T, D).contiguous().permute(0, 2, 1, 3) # B, T, N, D
        boxes_states = self.PIM(boxes_features) # B, T, N, D
        boxes_states = self.dropout(boxes_states)
        torch.cuda.empty_cache()

        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return {'activities':activities_scores}


class AT_volleyball(nn.Module):
    def __init__(self, cfg):
        super(AT_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # AT inference
        self.PE = Embfeature_PositionEmbedding(cfg = cfg, num_pos_feats = NFB//2)
        self.AT = Actor_Transformer(in_dim = NFB, temporal_pooled_first = cfg.temporal_pooled_first)
        self.fc_activities = nn.Linear(NFB, cfg.num_activities)
        self.fc_actions = nn.Linear(NFB, cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        # AT inference
        boxes_features = self.PE(boxes_features, boxes_in_flat)
        boxes_states = self.AT(boxes_features)
        torch.cuda.empty_cache()

        if self.cfg.temporal_pooled_first:
            # Predict actions
            actions_scores = self.fc_actions(boxes_states)
            actions_scores = actions_scores.view(B * N, -1)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim=1)
            boxes_states_pooled_flat = boxes_states_pooled.reshape(B, -1)
            activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num
        else:
            # Predict actions
            actions_scores = self.fc_actions(boxes_states)
            actions_scores = actions_scores.view((B, T, N, -1))
            actions_scores = torch.mean(actions_scores, dim = 1)
            actions_scores = actions_scores.view(B*N, -1)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim=1)
            boxes_states_pooled_flat = boxes_states_pooled.reshape(B*T, -1)
            activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num
            activities_scores = activities_scores.reshape(B, T, -1)
            activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        # # Predict actions
        # boxes_states_flat = boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores = self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        # # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)

        # return {'activities':activities_scores, 'actions':actions_scores, 'boxes_states':boxes_states.view((B, T, N, -1))} # [actions_scores, activities_scores] # activities_scores
        return {'activities':activities_scores}


class ARG_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(ARG_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph


        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False


        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # self.fc_actions = nn.Linear(NFG, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFG, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if not self.training:
            B = B * 3
            T = T // 3
            images_in.reshape((B, T) + images_in.shape[2:])
            boxes_in.reshape((B, T) + boxes_in.shape[2:])

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # GCN
        graph_boxes_features = boxes_features.reshape(B, T * N, NFG)

        #         visual_info=[]
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, boxes_in_flat)
        #             visual_info.append(relation_graph.reshape(B,T,N,N))

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, NFG)
        boxes_features = boxes_features.reshape(B, T, N, NFB)

        #         boxes_states= torch.cat( [graph_boxes_features,boxes_features],dim=3)  #B, T, N, NFG+NFB
        boxes_states = graph_boxes_features + boxes_features

        boxes_states = self.dropout_global(boxes_states)

        NFS = NFG

        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        # actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        # actions_scores = actions_scores.reshape(B, T, N, -1)
        # actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        if not self.training:
            B = B // 3
            # actions_scores = torch.mean(actions_scores.reshape(B, 3, N, -1), dim=1).reshape(B * N, -1)
            activities_scores = torch.mean(activities_scores.reshape(B, 3, -1), dim=1).reshape(B, -1)

        # return [activities_scores] # actions_scores, #'boxes_states':boxes_states
        # return {'activities':activities_scores, 'actions_scores':actions_scores}
        return {'activities':activities_scores}


class SACRF_BiUTE_volleyball(nn.Module):
    def __init__(self, cfg):
        super(SACRF_BiUTE_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # SACRF+BiUTE inference
        self.SACRF = SACRF(cfg, NFB, num_actions = cfg.num_actions)
        self.BiUTE = BiUTE(NFB, cfg.num_boxes)
        self.fc_activities = nn.Linear(NFB*2, cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)


        # AT inference
        action_scores, C_v, halt_loss = self.SACRF(boxes_features, boxes_in_flat)
        activities_feature = self.BiUTE(C_v)
        activities_scores = self.fc_activities(activities_feature)
        torch.cuda.empty_cache()

        action_scores = torch.mean(action_scores, dim = 1).view(B*N, -1)
        activities_scores = torch.mean(activities_scores, dim = 1).view(B, -1)

        return {'actions': action_scores, 'activities':activities_scores, 'halting':halt_loss, 'boxes_states':C_v}
        # return [actions_scores, activities_scores]  # activities_scores



class Dynamic_collective(nn.Module):
    def __init__(self, cfg):
        super(Dynamic_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        else:
            assert False
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        #self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        if not self.cfg.hierarchical_inference:
            self.DPI = Dynamic_Person_Inference(
                in_dim = in_dim,
                person_mat_shape = (T, N),
                stride = cfg.stride,
                kernel_size = cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio = cfg.sampling_ratio, # [1,2,4]
                group = cfg.group,
                scale_factor = cfg.scale_factor,
                beta_factor = cfg.beta_factor,
                parallel_inference = cfg.parallel_inference,
                cfg = cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape=(T, N),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg = cfg,)
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size=1, stride=1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        if self.cfg.lite_dim:
            boxes_features_all = boxes_features_all.permute(0, 3, 1, 2)
            boxes_features_all = self.point_conv(boxes_features_all)
            boxes_features_all = boxes_features_all.permute(0, 2, 3, 1)
            boxes_features_all = self.point_ln(boxes_features_all)
            boxes_features_all = F.relu(boxes_features_all, inplace = True)
        else:
            None

        # boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, NFB)
        # boxes_in = boxes_in.reshape(B, T, MAX_N, 4)

        #actions_scores = []
        activities_scores = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # 1,T,N,NFB
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4

            # Dynamic graph inference
            graph_boxes_features = self.DPI(boxes_features)
            torch.cuda.empty_cache()

            # cat graph_boxes_features with boxes_features
            boxes_states = graph_boxes_features + boxes_features  # 1, T, N, NFG
            boxes_states = boxes_states.permute(0, 2, 1, 3).view(N, T, -1)
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace=True)
            boxes_states = self.dropout_global(boxes_states)
            NFS = NFG
            # boxes_states = boxes_states.view(T, N, -1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim = 0)  # T, NFS
            acty_score = self.fc_activities(boxes_states_pooled)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)

        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        return {'activities':activities_scores}# activities_scores # actions_scores,
