import sys
sys.path.append(".")
from train_net_dynamic import *

cfg=Config('volleyball')
cfg.inference_module_name = 'higcin_volleyball'

cfg.device_list = "0"
cfg.use_gpu = True
cfg.use_multi_gpu = False
cfg.training_stage = 2
cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 1

# vgg16 setup
# cfg.backbone = 'vgg16'
# cfg.stage1_model_path = 'result/basemodel_VD_vgg16.pth'
# cfg.out_size = 22, 40
# cfg.emb_features = 512

# res18 setup
cfg.backbone = 'res18'
cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
cfg.out_size = 23, 40
cfg.emb_features = 512


# HIGCIN
cfg.crop_size = 7, 7

# Dynamic Inference setup
cfg.group = 1
cfg.stride = 1
cfg.ST_kernel_size = (3, 3)
cfg.dynamic_sampling = True
cfg.sampling_ratio = [1]  # [1,2,4]
cfg.lite_dim = None # 128
cfg.scale_factor = True
cfg.beta_factor = False
cfg.hierarchical_inference = False
cfg.parallel_inference = False

cfg.batch_size = 2
cfg.test_batch_size = 1
cfg.num_frames = 10
cfg.load_backbone_stage2 = True
cfg.train_learning_rate = 3e-5
cfg.lr_plan = {16: 1e-5} # {11: 3e-5, 21: 1e-5}
cfg.max_epoch = 30
cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

cfg.exp_note = 'HiGCIN Volleyball_stage2'
train_net(cfg)