import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection

from common.core.metric import PCKm, easy_calc_loss
from models.pose_resnet import PoseResNet

class ModsProxy(pl.LightningModule):

    @staticmethod
    def get_metrics(prefix=""):
        metrics = MetricCollection([
            PCKm(threshold=0.5)
        ],prefix=prefix)
        return metrics

    def __init__(self,config,_proxy=PoseResNet) -> None:
        super(ModsProxy,self).__init__()

        self.config = config
        self._model = _proxy(config)

        self.train_metrics = self.get_metrics(prefix="train_")
        self.valid_metrics = self.get_metrics(prefix="valid_")
        self.test_metrics = self.get_metrics(prefix="test_")

        print(f"{self.__class__.__name__} : loading model {self._model.__class__.__name__}")

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
        loss = easy_calc_loss(pred.float(),labels.float(),joints_vis.float())
        perf = self.train_metrics(pred,labels[:,-1,:,:,:])
        
        self.log_dict(perf,on_step=False,on_epoch=True,prog_bar=True)
        return {"loss": loss}

    @torch.no_grad()
    def _shared_eval_step(self,batch,metrics=None):
        image_path, center, scale, input_image, labels, joints, joints_vis = batch
        
        pred = self._model(input_image)
        return metrics(pred,labels[:,-1,:,:,:])

    def validation_step(self, batch, batch_idx):
        perf = self._shared_eval_step(batch,metrics=self.valid_metrics)
        self.log_dict(perf,on_step=False,on_epoch=True,prog_bar=True)
