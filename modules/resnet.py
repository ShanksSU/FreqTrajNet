import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from einops import rearrange
from .CoordAttention import CoordAtt
from .EMAttention import EMA

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
    
    def forward(self, x):
        # x: [B, N, C]
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        w_normed = F.normalize(w, dim=-2) 
        w_sq = w_normed ** 2

        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp) # b * h * n 
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class SpatialAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, clusters=1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.clusters = clusters
        self.tssa = AttentionTSSA(dim=input_dim, num_heads=num_heads)
        self.query_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim * clusters),
            nn.LayerNorm(input_dim * clusters)
        )
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, bias=True, batch_first=True)

    def forward(self, x):
        N, C, T, H, W = x.shape
        # [N*T, H*W, C]
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C).contiguous()
        x_enhanced = self.tssa(x_flat)
        gap_feature = x_enhanced.mean(dim=1) # [N*T, C]
        dynamic_query = self.query_generator(gap_feature).view(N * T, self.clusters, self.input_dim)
        attn_output, _ = self.mha(dynamic_query, x_enhanced, x_enhanced, need_weights=False)
        attn_output = attn_output.view(N, T, self.clusters, C).permute(0, 3, 1, 2).contiguous()
        
        return attn_output

class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size=9, window_stride=1, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(window_dilation, 1),
                                stride=(window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        N, C, T, H, W = x.shape
        x = x.view(N, C, T, H * W)
        x_unfolded = self.unfold(x) 
        x_reshaped = x_unfolded.view(N, C, self.window_size, T, H, W)
        
        return x_reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()

class FrequencyTemporalAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = input_size // 16
        self.conv_transform = nn.Conv1d(input_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv_back = nn.Conv1d(hidden_size, input_size, kernel_size=1, stride=1, padding=0)
        
        self.freq_att = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        # x: [N, C, T, H, W]
        out_pooled = x.mean(dim=[-2, -1]) # [N, C, T]
        out = self.conv_transform(out_pooled) # [N, hidden, T]
        orig_dtype = out.dtype
        out_f32 = out.to(torch.float32)
        freq_domain = torch.fft.rfft(out_f32, dim=2, norm='ortho')
        freq_mag = torch.abs(freq_domain) # [N, hidden, T/2+1]
        avg_freq = freq_mag.mean(dim=2).to(orig_dtype) # [N, hidden]
        max_freq = freq_mag.max(dim=2)[0].to(orig_dtype) # [N, hidden]
        att_weight = self.freq_att(avg_freq) + self.freq_att(max_freq)
        att_weight = torch.sigmoid(att_weight).unsqueeze(-1).to(torch.float32) # [N, hidden, 1]
        freq_domain_modulated = freq_domain * att_weight
        out_modified = torch.fft.irfft(freq_domain_modulated, n=out.size(2), dim=2, norm='ortho')
        out_modified = out_modified.to(orig_dtype)
        mask = self.conv_back(out_modified)
        mask = torch.sigmoid(mask).unsqueeze(-1).unsqueeze(-1) # [N, C, T, 1, 1]
        
        return x * (mask - 0.5) * self.alpha

class TrajectoryCorrelationModule(nn.Module):
    def __init__(self, channels, neighbors=3):
        super().__init__()
        self.neighbors = neighbors
        self.clusters = 1
        self.attpool = SpatialAttentionPooling(input_dim=channels, num_heads=1, clusters=self.clusters)
        self.weights_pool = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.unfold = UnfoldTemporalWindows(2 * self.neighbors + 1)
        self.weights_neighbor = nn.Parameter(torch.ones(self.neighbors * 2) / (self.neighbors * 2), requires_grad=True)
        
    def clustering(self, query, key):
        affinities = torch.einsum('bctp,bctl->btpl', query, key)
        weights = torch.sigmoid(affinities) - 0.5
        return torch.einsum('bctl,btpl->bctp', key, weights)

    def forward(self, x):
        N, C, T, H, W = x.shape
        x_mean = x.mean(3, keepdim=True).mean(4, keepdim=True) 
        x_max = x.flatten(3).max(-1)[0].view(N, C, T, 1, 1)    
        x_att = self.attpool(x).unsqueeze(-1)                  

        x_compressed = x_mean * self.weights_pool[0] + \
                       x_max  * self.weights_pool[1] + \
                       x_att  * self.weights_pool[2]
        x_compressed = x_compressed.squeeze(-1) # [N, C, T, 1]
        x2 = self.down_conv2(x)
        upfold_out = self.unfold(x2) # [N, C, T, Window, H, W]
        left = upfold_out[:, :, :, :self.neighbors, :, :]
        right = upfold_out[:, :, :, self.neighbors+1:, :, :]
        neighbors = torch.cat([left, right], dim=3) # [N, C, T, 2*Neighbors, H, W]
        w_n = self.weights_neighbor.view(1, 1, 1, -1, 1, 1)
        neighbors = neighbors * w_n
        neighbors_flat = neighbors.flatten(3) 
        features = self.clustering(x_compressed, neighbors_flat)
        
        return features.unsqueeze(-1) # [N, C, T, 1, 1]
        

class SpatialSalienceModule(nn.Module):
    def __init__(self, channels, reduction_channel):
        super().__init__()
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,1,1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,2,2), dilation=(1,2,2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,3,3), dilation=(1,3,3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.ema_attn = EMA(channels=reduction_channel, factor=32)
        # self.coord_attn = CoordAtt(reduction_channel, reduction_channel)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_down = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x_down) * self.weights[0] + \
                       self.spatial_aggregation2(x_down) * self.weights[1] + \
                       self.spatial_aggregation3(x_down) * self.weights[2]
        
        N, C, T, H, W = aggregated_x.size()
        x_2d = aggregated_x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        
        x_att = self.ema_attn(x_2d)
        
        aggregated_x = x_att.view(N, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        aggregated_x = self.conv_back(aggregated_x)

        return torch.sigmoid(aggregated_x) - 0.5

class SpatioTemporalCorrelationBlock(nn.Module):
    def __init__(self, channels, neighbors=3):
        super().__init__()
        reduction_channel = channels // 16
        self.correlation_module = TrajectoryCorrelationModule(channels, neighbors)
        self.identification_module = SpatialSalienceModule(channels, reduction_channel)

    def forward(self, x):
        E = self.correlation_module(x)
        M = self.identification_module(x)
        return E * M

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        # Layer 1
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        # Layer 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.corr2 = SpatioTemporalCorrelationBlock(self.inplanes, neighbors=1)
        self.temporal_weight2 = FrequencyTemporalAttention(self.inplanes)
        
        # Layer 3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr3 = SpatioTemporalCorrelationBlock(self.inplanes, neighbors=3)
        self.temporal_weight3 = FrequencyTemporalAttention(self.inplanes)
        
        # Layer 4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr4 = SpatioTemporalCorrelationBlock(self.inplanes, neighbors=5)
        self.temporal_weight4 = FrequencyTemporalAttention(self.inplanes)
        
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, 3, T, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        # Stage 2
        x = self.layer2(x) 
        x = x + self.corr2(x) * self.alpha[0]
        x = x + self.temporal_weight2(x)
        
        # Stage 3
        x = self.layer3(x)
        x = x + self.corr3(x) * self.alpha[1]
        x = x + self.temporal_weight3(x)
        
        # Stage 4
        x = self.layer4(x)
        x = x + self.corr4(x) * self.alpha[2]
        x = x + self.temporal_weight4(x)
    
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:]) 

        x = self.avgpool(x)
        x = x.flatten(1) # [N*T, C]
        x = self.dropout(x)
        x = self.fc(x)

        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    do_load_pretrained = kwargs.pop('pretrained', True)
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if do_load_pretrained:
        print("Loading ImageNet pretrained weights for ResNet18...")
        checkpoint = model_zoo.load_url(model_urls['resnet18'], map_location=torch.device('cpu'))
        layer_name = list(checkpoint.keys())
        for ln in layer_name :
            if 'conv' in ln or 'downsample.0.weight' in ln:
                checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint
        gc.collect()
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    do_load_pretrained = kwargs.pop('pretrained', True)
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if do_load_pretrained:
        print("Loading ImageNet pretrained weights for ResNet34...")
        checkpoint = model_zoo.load_url(model_urls['resnet34'], map_location=torch.device('cpu'))
        layer_name = list(checkpoint.keys())
        for ln in layer_name:
            if 'conv' in ln or 'downsample.0.weight' in ln:
                checkpoint[ln] = checkpoint[ln].unsqueeze(2)
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint
        gc.collect()
    return model