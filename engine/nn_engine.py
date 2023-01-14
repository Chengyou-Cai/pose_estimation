import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
class NNEngine():

    def __init__(self,config,_model_api,data_drive,monitor) -> None:
        super(NNEngine,self).__init__()
        
        self.config = config
        self._model_api = _model_api
        self.data_drive = data_drive

        self.logger = TensorBoardLogger(save_dir="./_logs/",name=_model_api._get_name()[:-3])
        self.model_ckpt = ModelCheckpoint(
            mode="min",
            save_top_k=3,
            save_last=True,
            monitor=monitor,
            dirpath=f"./_ckpt/{_model_api._get_name()[:-3]}/",
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
        self.trainer.fit(model=self._model_api,datamodule=self.data_drive)