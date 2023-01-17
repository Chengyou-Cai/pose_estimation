import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def set_environ(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICE._GPUS
    pl.seed_everything(config.HPARAM.RAND_SEED)


class SysEngine():

    def __init__(self,config,mods_proxy,data_drive,monitor="valid_PCKm") -> None:
        super(SysEngine,self).__init__()
        
        self.config = config
        self.mods_proxy = mods_proxy
        self.data_drive = data_drive

        self.logger = TensorBoardLogger(
            save_dir=f"./_logs/",
            name=mods_proxy._model._get_name()
            )
        self.model_ckpt = ModelCheckpoint(
            mode="min",
            save_top_k=3,
            save_last=True,
            monitor=monitor,
            dirpath=f"./_ckpt/{mods_proxy._model._get_name()}/",
            filename=f"{{epoch:02d}}_{{{monitor}:.2f}}"
            )
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            auto_select_gpus=True,
            min_epochs=1,
            max_epochs=self.config.HPARAM.MAX_EPOCHS,
            check_val_every_n_epoch=1,
            callbacks=[self.model_ckpt],
            logger=self.logger,
            gradient_clip_val=self.config.HPARAM.CLIP,
            profiler="simple"
        )
    
    def fit(self):
        print("start fitting...")
        self.data_drive.setup(stage='fit')
        self.trainer.fit(model=self.mods_proxy,datamodule=self.data_drive)