import torch
from torch import nn

"Efficient Multi-Scale Attention Module with Cross-Spatial Learning"

import torch
import torch.nn as nn

class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor if channels >= factor else channels
        assert channels % self.groups == 0, f"Channels {channels} must be divisible by groups {self.groups}"
        
        c_per_group = channels // self.groups
        
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.gn = nn.GroupNorm(c_per_group, c_per_group)
        self.conv1x1 = nn.Conv2d(c_per_group, c_per_group, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(c_per_group, c_per_group, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # (B, C, H, W)
        b, c, h, w = x.size()
        c_per_group = c // self.groups

        group_x = x.reshape(b * self.groups, c_per_group, h, w)  # (B*G, C/G, H, W)
        x_h = self.pool_h(group_x) # (B*G, C/G, H, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # (B*G, C/G, 1, W) -> (B*G, C/G, W, 1)
        
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # (B*G, C/G, H+W, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) 
        x2 = self.conv3x3(group_x) 

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) 
        x12 = x2.reshape(b * self.groups, c_per_group, -1)  
        y1 = torch.matmul(x11, x12) # (B*G, 1, H*W)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) 
        x22 = x1.reshape(b * self.groups, c_per_group, -1)  
        y2 = torch.matmul(x21, x22)  # (B*G, 1, H*W)
        weights = (y1 + y2).reshape(b * self.groups, 1, h, w)  
        weights_ = weights.sigmoid() 
        out = (group_x * weights_).reshape(b, c, h, w)
        
        return out

if __name__ == '__main__':
    # (B,C,H,W)
    input=torch.randn(1,512,7,7)
    Model = EMA(channels=512)
    output=Model(input)
    print(output.shape)
