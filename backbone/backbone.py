import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from thop import profile, clever_format
from utils import MAC2FLOP
from fvcore.nn import activation_count, flop_count, parameter_count

    
class MyInception_v3(nn.Module):
    def __init__(self,transform_input=False,pretrained=False):
        super(MyInception_v3,self).__init__()
        self.transform_input=transform_input
        inception=models.inception_v3(pretrained=pretrained)
        
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e

        # self.AuxLogits = inception.AuxLogits
        # self.Mixed_7a = inception.Mixed_7a
        # self.Mixed_7b = inception.Mixed_7b
        # self.Mixed_7c = inception.Mixed_7c
        
    def forward(self,x):
        outputs=[]
        
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        outputs.append(x)
        
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        # print(x.shape)
        outputs.append(x)

        # x = self.Mixed_7a(x)
        # x = self.Mixed_7b(x)
        # x = self.Mixed_7c(x)
        # print(x.shape)
        # outputs.append(x)
        
        return outputs
    

class MyVGG16(nn.Module):
    def __init__(self,pretrained=False):
        super(MyVGG16,self).__init__()
        
        vgg=models.vgg16(pretrained=pretrained)
     
        self.features=vgg.features
        
    def forward(self,x):
        x = self.features(x)
        #print(x.shape)
        return [x]
    
    
class MyVGG19(nn.Module):
    def __init__(self, pretrained = False):
        super(MyVGG19,self).__init__()
        
        vgg = models.vgg19(pretrained=pretrained)
     
        self.features = vgg.features
        
    def forward(self,x):
        x=self.features(x)
        return [x]


class MyRes18(nn.Module):
    def __init__(self, pretrained = False):
        super(MyRes18, self).__init__()
        res18 = models.resnet18(pretrained = pretrained)
        self.features = nn.Sequential(
            res18.conv1,
            res18.bn1,
            res18.relu,
            res18.maxpool,
            res18.layer1,
            res18.layer2,
            res18.layer3,
            res18.layer4
        )

    def forward(self, x):
        x = self.features(x)
        return [x]


class MyRes50(nn.Module):
    def __init__(self, pretrained=False):
        super(MyRes50, self).__init__()

        res50 = models.resnet50(pretrained=pretrained)

        self.features = nn.Sequential(
            res50.conv1,
            res50.bn1,
            res50.relu,
            res50.maxpool,
            res50.layer1,
            res50.layer2,
            res50.layer3,
            res50.layer4
        )

    def forward(self, x):
        x = self.features(x)
        return [x]

class MyAlex(nn.Module):
    def __init__(self, pretrained = False):
        super(MyAlex, self).__init__()

        alex = models.alexnet(pretrained = pretrained)

        self.features = alex.features

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        return [x]



if __name__=='__main__':
    None

