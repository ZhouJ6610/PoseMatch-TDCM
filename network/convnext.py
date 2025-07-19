import torch, timm
import torch.nn as nn

class convnext_v2(torch.nn.Module):
    def __init__(self):
        model = timm.create_model('convnextv2_large', pretrained=False)
        super().__init__()
        self.stem = model.stem          # Stem层
        self.stage1 = model.stages[0]   # Stage1 (包含Block和Downsample)
        self.norm = nn.BatchNorm2d(192)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.norm(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            # 仅初始化新增的BatchNorm层（跳过从预训练模型继承的层）
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    


class convnext_v2_tiny(torch.nn.Module):
    def __init__(self):
        model = timm.create_model('convnextv2_tiny', pretrained=False)
        super().__init__()
        self.stem = model.stem          # Stem层
        self.stage1 = model.stages[0]   # Stage1 (包含Block和Downsample)
        self.norm = nn.BatchNorm2d(96)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.norm(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            # 仅初始化新增的BatchNorm层（跳过从预训练模型继承的层）
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
            
class convnext_stage2(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('convnextv2_large', pretrained=False)

        self.stem = model.stem           # Stem 层
        self.stage1 = model.stages[0]    # Stage-1
        self.stage2 = model.stages[1]    # Stage-2（新增）
        self.norm = nn.BatchNorm2d(384)  # 对应 Stage-2 输出通道

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.norm(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)