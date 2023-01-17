import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.coco_dataset import COCODataset
import torchvision.transforms as transforms

class DataDrive(pl.LightningDataModule):

    def __init__(self,config,_drive=COCODataset) -> None:
        super(DataDrive,self).__init__()
        self.config = config
        self._drive = _drive # ?

    def prepare_data(self) -> None:
        pass

    def setup(self, stage="fit") -> None:
        assert (stage == 'fit' or stage == 'test')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        if stage == 'fit':
            self.train_set = self._drive(stage="train",config=self.config,transform=transform)
            print("!!! train set loaded !!!\n")
            self.valid_set = self._drive(stage="valid",config=self.config,transform=transform)
            print("!!! valid set loaded !!!\n")
        elif stage == 'test':
            self.test_set = self._drive(stage="test",config=self.config,transform=transform)
            print("!!! test set loaded !!!\n")

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