from typing import Type, Union
import os
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3

class ResNet(nn.Module):

    def __init__(self,block, layers, replace_stride_with_dilation= None) -> None:
        super(ResNet,self).__init__()

        self._norm_layer =  nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
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
        layers.append(
            block(
                inplanes=self.inplanes, 
                planes=planes, 
                stride=stride, 
                downsample=downsample, 
                dilation=previous_dilation, 
                norm_layer=norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

resnet_impl = {
    18: (BasicBlock, [2, 2, 2, 2],resnet.ResNet18_Weights),
    34: (BasicBlock, [3, 4, 6, 3],resnet.ResNet34_Weights),
    50: (Bottleneck, [3, 4, 6, 3],resnet.ResNet50_Weights),
    101: (Bottleneck, [3, 4, 23, 3],resnet.ResNet101_Weights),
    152: (Bottleneck, [3, 8, 36, 3],resnet.ResNet152_Weights)
    }

def implement(num_layers:int,pretrained:bool=False):
    assert num_layers in [18,34,50,101,152]
    block, layers, weights = resnet_impl[num_layers]
    model = ResNet(block,layers)
    
    if pretrained:
        os.environ['TORCH_HOME']="_ckpt/"
        m_param = model.state_dict()

        if num_layers==18:
            w_model = resnet.resnet18(weights)
        elif num_layers==34:
            w_model = resnet.resnet34(weights)
        elif num_layers==50:
            w_model = resnet.resnet50(weights)
        elif num_layers==101:
            w_model = resnet.resnet101(weights)
        elif num_layers==152:
            w_model = resnet.resnet152(weights)

        w_param = w_model.state_dict()
        w_param = {k: v for k, v in w_param.items() if k in m_param}
        m_param.update(w_param)
        model.load_state_dict(m_param)
    return model

    
    