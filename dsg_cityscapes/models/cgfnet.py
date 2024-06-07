import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, Activation, SegHead
from .backbone import Mobilenetv3



class CGFNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='mobilenet_v3', act_type='relu', use_aux=False):
        super(CGFNet, self).__init__()
        self.use_aux = use_aux
        self.base_block = BasicBlock(backbone_type)
        self.res_gate_attention1 = RGAM(in_channels=40, in_channels2=112, mid_channels=20)
        self.res_gate_attention2 = RGAM(in_channels=40, in_channels2=160, mid_channels=20)
        
        self.conv1 = conv1x1(40, 128)
        self.conv1_1 = conv1x1(160, 128)
        self.pappm = PAPPM(inplanes=160, branch_planes=40, outplanes=160)
        self.multi_attention_fusion = MAF(channels = 40, channels2=160, outchannels=128)
        
        # 分割头
        self.seg_head = SegHead(128, num_class, act_type)
        # 辅助分割头
        self.aux_head = SegHead(40, num_class, act_type)
        # 边界头
        # self.edge_head = SegHead(40, 1, act_type)
        

    def forward(self, x, is_training=False):
        # 保存原输入尺寸h, w
        size = x.size()[2:]
        # 下采样：/4，通道：24
        # 下采样：/8，通道：40
        # 下采样：/16，通道：112
        # 下采样：/32，通道：160
        x1, x2, x3, x4 = self.base_block(x)
        size2 = x2.size()[2:]
        identity = x2
        
        # 单向门控融合模块 /16→/8
        x2 = self.res_gate_attention1(x2, x3)

        # 单向门控融合模块 /32→/8
        x2 = self.res_gate_attention2(x2, x4) 

        # 多尺度特征提取模块
        x4 = self.pappm(x4)

        # 上采样到1/8尺寸
        x5 = F.interpolate(x4, size2, mode='bilinear', align_corners=True)

        # 高效多尺度注意力融合x2_coor和x5
        x_fusion = self.multi_attention_fusion(x2, x5)
         
        # sum融合
        # x2 = self.conv1(x2)
        # x5 = self.conv1_1(x5)
        # x_fusion = x2 + x5

        # 分割头
        x = self.seg_head(x_fusion)
        # 输出高分辨率特征图    
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        # 推理时可丢弃辅助头
        if is_training and self.use_aux:
            # 辅助头
            x_aux = self.aux_head(identity) 
            return x, (x_aux,)
        else:
            return x


class BasicBlock(nn.Module):
    def __init__(self, backbone_type):
        super(BasicBlock, self).__init__()
        if 'mobilenet_v3' in backbone_type:
            self.backbone = Mobilenetv3()
        else:
            raise ValueError('Backbone type is not supported.')
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        return x1, x2, x3, x4


class RGAM(nn.Module):
    def __init__(self, in_channels, in_channels2, mid_channels, act_type='relu'):
        super(RGAM, self).__init__()
        self.relu = nn.ReLU()  
        self.conv3x3 = nn.Sequential(
                                nn.ReLU(),
                                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(mid_channels)                  
                                )
        self.conv1x1 = nn.Sequential(
                                nn.ReLU(),
                                nn.Conv2d(in_channels2, mid_channels, kernel_size=1, bias=False),
                                nn.BatchNorm2d(mid_channels)                  
                                )
        self.up = nn.Sequential(
                                nn.Conv2d(in_channels2, in_channels, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_channels)
                                )
    def forward(self, x, y):
        size = x.size()[2:]
        residual = x

        y_q = self.conv1x1(y)
        y_q = F.interpolate(y_q, size=size, mode='bilinear', align_corners=False)
        x_k = self.conv3x3(x)
        edge_map = torch.sigmoid(torch.sum(y_q * x_k, dim=1).unsqueeze(1))
        y = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
        y = self.up(y)
        x = (1-edge_map)*x + edge_map*y
        
        x += residual
        x = self.relu(x)
        return x

class DAPPM(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super(DAPPM, self).__init__()
        hid_channels = int(in_channels // 4)
        
        self.conv0 = ConvBNAct(in_channels, out_channels, 1, act_type=act_type)
        self.conv1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.pool2 = self._build_pool_layers(in_channels, hid_channels, 5, 2)
        self.conv2 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool3 = self._build_pool_layers(in_channels, hid_channels, 9, 4)
        self.conv3 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool4 = self._build_pool_layers(in_channels, hid_channels, 17, 8)
        self.conv4 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool5 = self._build_pool_layers(in_channels, hid_channels, -1, -1)
        self.conv5 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.conv_last = ConvBNAct(hid_channels*5, out_channels, 1, act_type=act_type)
        
    def _build_pool_layers(self, in_channels, out_channels, kernel_size, stride):
        layers = []
        if kernel_size == -1:
            layers.append(nn.AdaptiveAvgPool2d(1))
        else:
            padding = (kernel_size - 1) // 2
            layers.append(nn.AvgPool2d(kernel_size, stride, padding))
        layers.append(conv1x1(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()[2:]
        y0 = self.conv0(x)
        y1 = self.conv1(x)
        
        y2 = self.pool2(x)
        y2 = F.interpolate(y2, size, mode='bilinear', align_corners=True)
        y2 = self.conv2(y1 + y2)
        
        y3 = self.pool3(x)
        y3 = F.interpolate(y3, size, mode='bilinear', align_corners=True)
        y3 = self.conv3(y2 + y3)
        
        y4 = self.pool4(x)
        y4 = F.interpolate(y4, size, mode='bilinear', align_corners=True)
        y4 = self.conv4(y3 + y4)
    
        y5 = self.pool5(x)
        y5 = F.interpolate(y5, size, mode='bilinear', align_corners=True)
        y5 = self.conv5(y4 + y5)
        
        x = self.conv_last(torch.cat([y1, y2, y3, y4, y5], dim=1)) + y0
    
        return x

class PAPPM(nn.Module):
    """
    Args:
        inplanes (int): 输入特征图的通道数
        branch_planes (int): 分支网络的通道数
        outplanes (int): 输出特征图的通道数
        BatchNorm (nn.Module, optional): 使用的归一化层，默认为 nn.BatchNorm2d. Defaults to nn.BatchNorm2d.
    
    Returns:
        None
    
    功能：定义PAPPM模块，包含多个尺度的特征提取和融合。
    """
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )


    def forward(self, x):
        width = x.shape[-1] # N_W倒数一个元素
        height = x.shape[-2] # N_H倒数第二个元素
        scale_list = []
        algc = False
        x_ = self.scale0(x) # 1x1卷积 out：12
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_) # N out_channels: branch_channels
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_) # N out_channels: branch_channels        
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_) # N out_channels: branch_channels
        
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_) # N out_channels: branch_channels
        
        scale_out = self.scale_process(torch.cat(scale_list, 1)) # 3x3卷积 out_channels: 4 * branch_channels
       
        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out
    
class MAF(nn.Module):
    def __init__(self, channels, channels2, outchannels, factor=8):
        """
        EMA模块的构造函数。
        
        Args:
            channels (int): 输入张量的通道数。
            channels2 (int): 另一个输入张量的通道数。
            outchannels (int): 输出张量的通道数。
            factor (int, optional): 分组数量，用于特征分组，默认为8。
        
        Returns:
            None
        
        """
        super(MAF, self).__init__()
        # 设置分组数量：8，用于特征分组
        self.groups = factor
        # 确保分组后的通道数大于0
        assert outchannels // self.groups > 0
        self.conv = nn.Sequential(
                                nn.ReLU(),
                                nn.Conv2d(channels+channels2, outchannels, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(outchannels)                  
                                )
        # softmax激活函数，用于归一化
        self.softmax = nn.Softmax(-1)
        # 全局平均池化，生成通道描述符
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # 水平方向的平均池化，用于编码水平方向的全局信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向的平均池化，用于编码垂直方向的全局信息
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # GroupNorm归一化，减少内部协变量偏移
        self.gn = nn.GroupNorm(outchannels // self.groups, outchannels // self.groups)
        # 1x1卷积，用于学习跨通道的特征
        self.conv1x1 = nn.Conv2d(outchannels // self.groups, outchannels // self.groups, kernel_size=1, stride=1, padding=0)
        # 3x3卷积，用于捕捉更丰富的空间信息
        self.conv3x3 = nn.Conv2d(outchannels // self.groups, outchannels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        b, c, h, w = x.size()
        # 对输入特征图进行分组处理
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # 应用水平和垂直方向的全局平均池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # 通过1x1卷积和sigmoid激活函数，获得注意力权重
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # 应用GroupNorm和注意力权重调整特征图
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        # 将特征图通过全局平均池化和softmax进行处理，得到权重
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # 通过矩阵乘法和sigmoid激活获得最终的注意力权重，调整特征图
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # 将调整后的特征图重塑回原始尺寸
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)