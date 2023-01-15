import torch
import torch.nn as nn

from models.component.resnet import implement as resnet_impl

class PoseResNet(nn.Module):

    def __init__(self,config) -> None:
        super(PoseResNet,self).__init__()
        self.config = config

        self.resnet = resnet_impl(
            num_layers=config.MODEL.NUM_LAYERS,
            pretrained=config.MODEL.PRETRAINED
            )

        self.inplanes = self.resnet.inplanes
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            config.MODEL.NUM_DECONV_LAYERS,
            config.MODEL.NUM_DECONV_FILTERS,
            config.MODEL.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=config.MODEL.NUM_DECONV_FILTERS[-1],
            out_channels=config.DATA.NUM_JOINTS,
            kernel_size=config.MODEL.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if config.MODEL.FINAL_CONV_KERNEL == 3 else 0
        )

    def _get_deconv_cfg(self, deconv_kernel):
        # deconv_kernel:(padding,output_padding)
        deconvs = {
            4:(1,0),
            3:(1,1),
            2:(0,0)
        }
        padding, output_padding = deconvs[deconv_kernel]

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

import torch.optim as optim
import pytorch_lightning as pl
from common.core.metric import easy_calc_loss 
class PoseResNetApi(pl.LightningModule):

    def __init__(self,config) -> None:
        super(PoseResNetApi,self).__init__()

        self.config = config
        self._model = PoseResNet(config)

        print(f"{self.__class__.__name__} : loading model")

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self._model.parameters(),
            lr=self.config.HPARAM.LR,
            weight_decay=self.config.HPARAM.WD
            )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            self.config.HPARAM.LR_STEP, 
            self.config.HPARAM.LR_FACTOR
            )
        return {'optimizer':optimizer,'lr_scheduler': scheduler}
    
    def training_step(self, batch, batch_idx):
        image_path, center, scale, input_image, labels, joints, joints_vis = batch
        
        pred = self._model(input_image)
        loss = easy_calc_loss(pred,labels,joints_vis)

        self.log_dict({"loss":loss},on_step=False,on_epoch=True,prog_bar=True)
        return {"loss": loss}

    @torch.no_grad()
    def _shared_eval_step(self,batch,metrics=None):
        image_path, center, scale, input_image, labels, joints, joints_vis = batch
        pred = self._model(input_image)
        perf = metrics(pred,labels,joints_vis)
        return perf

    def validation_step(self, batch, batch_idx):
        perf = self._shared_eval_step(batch)
        self.log_dict(perf,on_step=False,on_epoch=True,prog_bar=True)