import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')

cfg.use_multi_gpu = False
cfg.device_list="0"
cfg.training_stage=1
cfg.stage1_model_path=''
cfg.train_backbone=True
cfg.test_before_train = True

# VGG16
cfg.backbone = 'vgg16'
cfg.image_size = 720, 1280
cfg.out_size = 22, 40
cfg.emb_features = 512

cfg.num_before = 10
cfg.num_after = 9

cfg.batch_size=8
cfg.test_batch_size=1
cfg.num_frames=1
cfg.train_learning_rate=1e-5
cfg.lr_plan={}
cfg.max_epoch=200
cfg.set_bn_eval = False
cfg.actions_weights=[[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]  

cfg.exp_note='Volleyball_stage1'
train_net(cfg)