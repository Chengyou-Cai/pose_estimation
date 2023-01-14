import os

import torchvision.models.resnet as resnet


class ResNet(resnet.ResNet):

    def __init__(self,block, layers, replace_stride_with_dilation= None) -> None:
        super(ResNet,self).__init__(
            block = block,layers = layers,
            replace_stride_with_dilation = \
                replace_stride_with_dilation
            )
        delattr(self,"avgpool")
        delattr(self,"fc")

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

        return x

    def forward(self, x):
        return self._forward_impl(x)

resnet_impl = {
    18: (resnet.BasicBlock, [2, 2, 2, 2],resnet.ResNet18_Weights),
    34: (resnet.BasicBlock, [3, 4, 6, 3],resnet.ResNet34_Weights),
    50: (resnet.Bottleneck, [3, 4, 6, 3],resnet.ResNet50_Weights),
    101: (resnet.Bottleneck, [3, 4, 23, 3],resnet.ResNet101_Weights),
    152: (resnet.Bottleneck, [3, 8, 36, 3],resnet.ResNet152_Weights)
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

    
    