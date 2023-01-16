
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DataDrive(pl.LightningDataModule):

    def __init__(self,config,_drive=None) -> None:
        super(DataDrive,self).__init__()
        self.config = config
        self._drive = _drive # ?

    def prepare_data(self) -> None:
        pass

    def setup(self, stage="fit") -> None:
        assert (stage == 'fit' or stage == 'test')
        if stage == 'fit':
            self.train_set = self._drive["train"]
            self.valid_set = self._drive["valid"]
        elif stage == 'test':
            self.test_set = self._drive["test"]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            shuffle=False,
            batch_size=self.config.HPARAM.BATCH_SIZE,
            num_workers=self.config.HPARAM.NUM_WORKERS,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            shuffle=False,
            batch_size=self.config.HPARAM.BATCH_SIZE,
            num_workers=self.config.HPARAM.NUM_WORKERS,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            shuffle=False,
            batch_size=self.config.HPARAM.BATCH_SIZE,
            num_workers=self.config.HPARAM.NUM_WORKERS,
            pin_memory=True
        )