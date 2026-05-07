"""
Adapted from `https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py`
"""
from argparse import Namespace
from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import Tensor
import torch.nn as nn

def conv1x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1d(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock1d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck1d(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1d(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock1d, Bottleneck1d]],
        layers: List[int],
        num_lead: int,
        backbone_out_dim: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(num_lead, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, backbone_out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1d):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock1d):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self, 
        block: Type[Union[BasicBlock1d, Bottleneck1d]],
        planes: int, 
        blocks: int,   
        stride: int = 1, 
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): Tensor of shape (bs, n_lead=1, seqlen).

        Returns:
            Tensor: _description_
        """
        return self._forward_impl(x)

class ResNet1dDecoder(nn.Module):

    def __init__(
        self,
        seqlen: int, 
        z_dim: int, 
        h_dim: int,
        chunk_len: int
    ) -> None:
        super(ResNet1dDecoder, self).__init__()
        n_chunk = seqlen // chunk_len
        self.h_dim = h_dim
        self.fc_decode = nn.Linear(z_dim, h_dim * (n_chunk // 4))
        self.deconv1 = nn.ConvTranspose1d(
            h_dim, h_dim//2, kernel_size=4, stride=2, padding=1)
        self.resblock1 = BasicBlock1d(h_dim//2, h_dim//2)

        self.deconv2 = nn.ConvTranspose1d(
            h_dim//2, h_dim//4, kernel_size=4, stride=2, padding=1)
        self.resblock2 = BasicBlock1d(h_dim//4, h_dim//4)

        # self.deconv3 = nn.ConvTranspose1d(
        #     h_dim//4, h_dim//8, kernel_size=4, stride=2, padding=1)
        # self.resblock3 = BasicBlock1d(h_dim//8, h_dim//8)
        self.fc_linear = nn.Linear(h_dim//4, chunk_len)

    def forward(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), self.h_dim, -1)
        x = self.deconv1(x)
        x = self.resblock1(x)
        x = self.deconv2(x)
        x = self.resblock2(x)
        x = x.transpose(1, 2)
        x = self.fc_linear(x)
        x = x.view(x.size(0), -1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x
    
class ResNet1dDecoder02(nn.Module):

    def __init__(
        self,
        seqlen: int, 
        z_dim: int, 
        h_dim: int,
        chunk_len: int
    ) -> None:
        super(ResNet1dDecoder02, self).__init__()
        n_chunk = seqlen // chunk_len
        self.h_dim = h_dim
        self.fc_decode = nn.Linear(z_dim, h_dim * (n_chunk // 4))
        self.deconv1 = nn.ConvTranspose1d(
            h_dim, h_dim//2, kernel_size=4, stride=2, padding=1)
        self.resblock1 = BasicBlock1d(h_dim//2, h_dim//2)

        self.deconv2 = nn.ConvTranspose1d(
            h_dim//2, h_dim//4, kernel_size=4, stride=2, padding=1)
        self.resblock2 = BasicBlock1d(h_dim//4, h_dim//4)

        self.deconv3 = nn.ConvTranspose1d(
            h_dim//4, h_dim//8, kernel_size=4, stride=2, padding=1)
        self.resblock3 = BasicBlock1d(h_dim//8, h_dim//8)
        self.fc_linear = nn.Linear(h_dim//8, chunk_len // 2)

    def forward(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), self.h_dim, -1)
        x = self.deconv1(x)
        x = self.resblock1(x)
        x = self.deconv2(x)
        x = self.resblock2(x)
        x = self.deconv3(x)
        x = self.resblock3(x)
        x = x.transpose(1, 2)
        x = self.fc_linear(x)
        x = x.view(x.size(0), -1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

class ResNet1dDecoder03(nn.Module):

    def __init__(
        self,
        seqlen: int, 
        z_dim: int, 
        h_dim: int,
        chunk_len: int
    ) -> None:
        super(ResNet1dDecoder03, self).__init__()
        n_chunk = seqlen // chunk_len
        self.h_dim = h_dim
        self.fc_decode = nn.Linear(z_dim, h_dim * (n_chunk // 4))
        self.deconv1 = nn.ConvTranspose1d(
            h_dim, h_dim//2, kernel_size=4, stride=2, padding=1)
        self.resblock1 = BasicBlock1d(h_dim//2, h_dim//2)

        self.deconv2 = nn.ConvTranspose1d(
            h_dim//2, h_dim//4, kernel_size=4, stride=2, padding=1)
        self.resblock2 = BasicBlock1d(h_dim//4, h_dim//4)

        self.deconv3 = nn.ConvTranspose1d(
            h_dim//4, h_dim//4, kernel_size=4, stride=2, padding=1)
        self.resblock3 = BasicBlock1d(h_dim//4, h_dim//4)
        self.fc_linear = nn.Linear(h_dim//4, h_dim//4)
        self.fc_last = nn.Linear(h_dim//4, chunk_len // 2)

    def forward(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), self.h_dim, -1)
        x = self.deconv1(x)
        x = self.resblock1(x)
        x = self.deconv2(x)
        x = self.resblock2(x)
        x = self.deconv3(x)
        x = self.resblock3(x)
        x = x.transpose(1, 2)
        x = self.fc_linear(x)
        x = self.fc_last(x)
        x = x.view(x.size(0), -1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x


class ResNet18(nn.Module):
    
    def __init__(self, params: Namespace, add_clf: bool = False, with_sigmoid: bool = True):
        super(ResNet18, self).__init__()

        block = BasicBlock1d
        layers = [2, 2, 2, 2]
        self.backbone = ResNet1d(
            block, layers, num_lead=params.num_lead, 
            backbone_out_dim=params.enc_out_dim
        )
        self.add_clf = add_clf
        if add_clf:
            self.clf = nn.Linear(params.enc_out_dim, 1)
            self.sigmoid = nn.Sigmoid() if with_sigmoid else None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [bs, seqlen] or [bs, n_lead, seqlen]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            pass
        else:
            raise

        x = self.backbone(x)
        if self.add_clf:
            x = self.clf(x)
            x = x.squeeze(1)
            if self.sigmoid is not None:
                x = self.sigmoid(x)
        return x

# class ResNet34(nn.Module):
    
#     def __init__(self, params: Namespace):
#         super(ResNet34, self).__init__()

#         block = BasicBlock1d
#         layers = [3, 4, 6, 3]
#         self.backbone = ResNet1d(
#             block, layers, num_lead=params.num_lead, 
#             backbone_out_dim=params.backbone_out_dim
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         x = x.unsqueeze(1)
#         return self.backbone(x)

# class ResNet50(nn.Module):
    
#     def __init__(self, params: Namespace):
#         super(ResNet50, self).__init__()

#         block = Bottleneck1d
#         layers = [3, 4, 6, 3]
#         self.backbone = ResNet1d(
#             block, layers, num_lead=params.num_lead, 
#             backbone_out_dim=params.backbone_out_dim
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         x = x.unsqueeze(1)
#         return self.backbone(x)