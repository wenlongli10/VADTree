"""
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RepBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, kernel_size=3, inference_mode=False):

        super(RepBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = 1
        self.identity = (stride == 1) and (in_planes == planes)
        self.inference_mode = inference_mode

        if self.inference_mode:
            self.rep_block = nn.Conv2d(
                in_channels=in_planes,
                out_channels=planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2
            )
        else:
            self.cb1 = nn.Sequential()
            self.cb1.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size//2,
                    bias=False
                ),
            )
            self.cb1.add_module("bn", nn.BatchNorm2d(planes))

            self.cb2 = nn.Sequential()
            self.cb2.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
            )
            self.cb2.add_module("bn", nn.BatchNorm2d(num_features=planes))

            if self.identity:
                self.shortcut = nn.BatchNorm2d(num_features=planes)

    def forward(self, x):
        if self.inference_mode:
            out = self.rep_block(x)
        else:
            out = self.cb1(x) + self.cb2(x)
            if self.identity:
                out += self.shortcut(x)
        out = F.relu(out)
        return out

    def fuse_conv_bn(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            conv_bias = branch.conv.bias
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            input_dim = self.in_planes // self.groups
            kernel_value = np.zeros((self.in_planes, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_planes):
                kernel_value[i, i % input_dim, 1, 1] = 1
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            conv_bias = None
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        weight = kernel * t
        if (conv_bias is not None):
            bias = beta - running_mean * gamma / std + gamma / std * conv_bias
        else:
            bias = beta - running_mean * gamma / std
        return weight, bias

    def reparameterize(self):
        if self.inference_mode:
            return

        k3, b3 = self.fuse_conv_bn(self.cb1)
        k1, b1 = self.fuse_conv_bn(self.cb2)
        kernel, bias = k3 + F.pad(k1, (1, 1, 1, 1)), b3 + b1
        if self.identity:
            ki, bi = self.fuse_conv_bn(self.shortcut)
            kernel += ki
            bias += bi

        self.rep_block = nn.Conv2d(self.in_planes, self.planes, self.kernel_size,
                                   stride=self.stride, padding=self.kernel_size//2)
        
        self.rep_block.weight.data = kernel
        self.rep_block.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("cb1")
        self.__delattr__("cb2")
        if self.identity:
            self.__delattr__("shortcut")

        self.inference_mode = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes, affine=True),
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_planes, decoder_hidden, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_planes, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, decoder_hidden, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RepVGG(nn.Module):
    def __init__(self, in_planes, decoder_hidden, block, num_blocks, num_classes=10):
        super(RepVGG, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_planes, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, decoder_hidden, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def ResNetXX(in_planes, decoder_hidden, resnet_type=1):
    type_map = {1: [1, 1, 1, 1], 2: [2, 2, 2, 2], 3: [3, 4, 6, 3]}
    return ResNet(in_planes, decoder_hidden, BasicBlock, type_map[resnet_type])


def RepVGG8(in_planes, decoder_hidden):
    return RepVGG(in_planes, decoder_hidden, RepBlock, [2, 2, 2, 2])


def ResNet18(in_planes, decoder_hidden):
    return ResNet(in_planes, decoder_hidden, BasicBlock, [1, 1, 1, 1])


def ResNet34(in_planes, decoder_hidden):
    return ResNet(in_planes, decoder_hidden, BasicBlock, [3, 4, 6, 3])


def ResNet50(in_planes, decoder_hidden):
    return ResNet(in_planes, decoder_hidden, Bottleneck, [3, 4, 6, 3])


def ResNet101(in_planes, decoder_hidden):
    return ResNet(in_planes, decoder_hidden, Bottleneck, [3, 4, 23, 3])


def ResNet152(in_planes, decoder_hidden):
    return ResNet(in_planes, decoder_hidden, Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18(27, 512)
    y = net(torch.randn(4, 27, 11, 11))
    print(y.size())

if __name__ == "__main__":
    test()