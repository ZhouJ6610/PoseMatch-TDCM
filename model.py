import torch 
import torch.nn as nn
import torch.nn.functional as F
from network import convnext_v2
from utils import weight_init


class Model(nn.Module):
    def __init__(self, snapshot, device):
        super(Model, self).__init__()
        self.snapshot       =   snapshot
        self.c              =   192
        self.device = device
        self.backbone_T     =   convnext_v2().to(device)
        self.backbone_O     =   convnext_v2().to(device)
        self.pointwise_conv =   nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, padding=1).to(device) 
        self.conv1          =   nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=3, padding=1).to(device)
        self.conv2          =   nn.Conv2d(in_channels=self.c//2, out_channels=self.c, kernel_size=3, padding=1).to(device)
        self.upsample       =   nn.PixelShuffle(2).to(device) 
        self.conv_score     =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1).to(device),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=self.c//16, kernel_size=3, padding=1).to(device),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//16, out_channels=1, kernel_size=1, padding=0).to(device),
                                    nn.Sigmoid()
                                )  
        self.conv_sign      =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1).to(device),
                                    nn.BatchNorm2d(self.c//8),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=self.c//16, kernel_size=3, padding=1).to(device),
                                    nn.BatchNorm2d(self.c//16),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//16, out_channels=1, kernel_size=1, padding=0).to(device),
                                    nn.Sigmoid()
                                )
        self.conv_cos       =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1).to(device),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=self.c//16, kernel_size=3, padding=1).to(device),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//16, out_channels=1, kernel_size=1, padding=0).to(device),
                                    nn.Tanh()
                                )
        self.conv_scale_x   =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1).to(device),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=self.c//16, kernel_size=3, padding=1).to(device),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//16, out_channels=1, kernel_size=1, padding=0).to(device),
                                    nn.Sigmoid()
                                )
        self.conv_scale_y   =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1).to(device),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=self.c//16, kernel_size=3, padding=1).to(device),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//16, out_channels=1, kernel_size=1, padding=0).to(device),
                                    nn.Sigmoid()
                                )
        self.bn1            =   nn.BatchNorm2d(self.c*2)
        self.bn2            =   nn.BatchNorm2d(self.c)
        self.sigmoid        =   nn.Sigmoid()

        self.initialize()

    def forward(self, origin, template):
        origin, template = origin.to(self.device), template.to(self.device)

        # W=H = 56, K = 4
        stage1_o = self.backbone_O(origin)       # (B, self.c, 56, 56)
        stage1_t = self.backbone_T(template)       # (B, self.c, 9, 9)
                        
        # depthiwise + pointwise(backbone_T更新梯度)
        output = []
        b, c, th, tw = stage1_t.shape
        for i in range(b):
            out = F.conv2d(
                    input=stage1_o[i].unsqueeze(0), # [self.c, 1, 56, 56]
                    weight=stage1_t[i].view(c, 1, th, tw), # 显式重塑权重,[self.c, 1, 9, 9]
                    stride=1, padding=(th//2, tw//2),
                    groups=self.c
                    )
            out = self.pointwise_conv(out)
            output.append(out)
        feature = torch.cat(output, dim=0)
        # print("feature: ", feature.shape)
        
        out_conv1 = F.relu(self.bn1(self.conv1(feature)))         # (B, self.c*2, 49, 49) 
        out_subconv1 = self.upsample(out_conv1)                   # (B, self.c//2, 98, 98)                                                                                      
        out_conv2 = F.relu(self.bn2(self.conv2(out_subconv1)))    # (B, self.c, 98, 98)
        out = self.upsample(out_conv2)                            # (B, self.c//4 , 196 , 196)

        score    = self.conv_score(out)       # (B, 1, 196, 196)
        sign     = self.conv_sign(out)
        cos      = self.conv_cos(out)
        scale_x  = self.conv_scale_x(out) * 1.5 + 0.5       # scale: 0.5 ~ 2
        scale_y  = self.conv_scale_y(out) * 1.5 + 0.5       # scale: 0.5 ~ 2
                
        return score, sign, cos, scale_x, scale_y
        

    def initialize(self):
        if self.snapshot:
            print('load model...')
            self.load_state_dict(torch.load(self.snapshot))
        else:
            weight_init(self)
            nn.init.kaiming_normal_(self.pointwise_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.pointwise_conv.bias, 0)