import sys
sys.path.append(".")
from train_net_dynamic import *

cfg=Config('collective')
cfg.inference_module_name = 'dynamic_collective'

cfg.device_list="0"
cfg.training_stage=2
cfg.use_gpu = True
cfg.use_multi_gpu = False
cfg.train_backbone = True
cfg.load_backbone_stage2 = True

# ResNet18
cfg.backbone = 'res18'
cfg.image_size = 480, 720
cfg.out_size = 15, 23
cfg.emb_features = 512
cfg.stage1_model_path = 'result/basemodel_CAD_res18.pth'

# VGG16
# cfg.backbone = 'vgg16'
# cfg.image_size = 480, 720
# cfg.out_size = 15, 22
# cfg.emb_features = 512
# cfg.stage1_model_path = 'result/basemodel_CAD_vgg16.pth'

cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10
cfg.num_graph = 4
cfg.tau_sqrt=True
cfg.batch_size = 2
cfg.test_batch_size = 8
cfg.test_interval_epoch = 1
cfg.train_learning_rate = 5e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-4
cfg.lr_plan = {}
cfg.max_epoch = 30


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

cfg.exp_note='Dynamic_collective'
train_net(cfg)