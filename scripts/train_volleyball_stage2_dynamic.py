import sys
sys.path.append(".")
from train_net_dynamic import *

cfg=Config('volleyball')
cfg.inference_module_name = 'dynamic_volleyball'

cfg.device_list = "0,1"
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 1

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.stage1_model_path = 'result/basemodel_VD_vgg16.pth'
cfg.out_size = 22, 40
cfg.emb_features = 512

# res18 setup
# cfg.backbone = 'res18'
# cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
# cfg.out_size = 23, 40
# cfg.emb_features = 512

# Dynamic Inference setup
cfg.group = 1
cfg.stride = 1
cfg.ST_kernel_size = [(3, 3)] #[(3, 3),(3, 3),(3, 3),(3, 3)]
cfg.dynamic_sampling = True
cfg.sampling_ratio = [1]
cfg.lite_dim = 128 # None # 128
cfg.scale_factor = True
cfg.beta_factor = False
cfg.hierarchical_inference = False
cfg.parallel_inference = False
cfg.num_DIM = 1
cfg.train_dropout_prob = 0.3

cfg.batch_size = 2
cfg.test_batch_size = 1
cfg.num_frames = 10
cfg.load_backbone_stage2 = True
cfg.train_learning_rate = 1e-4
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.max_epoch = 60
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
cfg.lr_plan = {11: 1e-5}
cfg.max_epoch = 30
cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

cfg.exp_note = 'Dynamic Volleyball_stage2_res18_litedim128_reproduce_1'
train_net(cfg)
