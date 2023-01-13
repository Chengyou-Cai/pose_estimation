import torch
import torch.nn as nn

from models.component.resnet import implement as resnet_impl

class PoseResNet():

    def __init__(self,config) -> None:
        self.config = config

        self.resnet = resnet_impl(
            num_layers=config.MODEL.NUM_LAYERS,
            pretrained=config.MODEL.PRETRAINED
            )
                # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            config.MODEL.NUM_DECONV_LAYERS,
            config.MODEL.NUM_DECONV_FILTERS,
            config.MODEL.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=config.MODEL.NUM_DECONV_FILTERS[-1],
            out_channels=config.MODEL.NUM_JOINTS,
            kernel_size=config.MODEL.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if config.MODEL.FINAL_CONV_KERNEL == 3 else 0
        )

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.config.MODEL.DECONV_WITH_BIAS))
            layers.append(nn.BatchNorm2d(planes, momentum=self.config.HPARAM.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x