# Channel Separated Convolutional Network (CSN) as presented in Video Classification with Channel-Separated Convolutional Networks(https://arxiv.org/pdf/1904.02811v4.pdf)
# replace 3x3x3 convolution with 1x1x1 conv + 3x3x3 depthwise convolution (ip) or with 3x3x3 depthwise convolution (ir)

import torch
import torch.nn as nn

        
class CSNBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, channels, stride=1, mode='ip'):
        super().__init__()
        
        assert mode in ['ip', 'ir']
        self.mode = mode
        
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        conv2 = []
        if self.mode == 'ip':
            conv2.append(nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False))
        conv2.append(nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=channels))
        self.conv2 = nn.Sequential(*conv2)
        self.bn2 = nn.BatchNorm3d(channels)
        
        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )
        
    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
            
        out += shortcut
        out = self.relu(out)
        
        return out


class CSN(nn.Module):
    def __init__(self, block, layers, num_classes, mode='ip'):
        super().__init__()
        
        assert mode in ['ip', 'ir']
        self.mode = mode
        
        self.in_channels = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride, mode=self.mode))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels, mode=self.mode))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


def csn26(num_classes, mode='ip'):
    return CSN(CSNBottleneck, [1,2,4,1], num_classes=num_classes, mode=mode)


def csn50(num_classes, mode='ip'):
    return CSN(CSNBottleneck, [3,4,6,3], num_classes=num_classes, mode=mode)


def csn101(num_classes, mode='ip'):
    return CSN(CSNBottleneck, [3,4,23,3], num_classes=num_classes, mode=mode)


def csn152(num_classes):
    return CSN(CSNBottleneck, [3,8,36,3], num_classes=num_classes)
    