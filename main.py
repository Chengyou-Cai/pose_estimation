from engine.data_drive import DataDrive
from engine.net_engine import NetEngine, set_environ
from myexps.coco.res.config import Config as COCOResConfig

from datasets.coco_dataset import COCODataset
from models.pose_resnet import PoseResNetApi


def main():

    config = COCOResConfig()
    set_environ(config=config)

    

if __name__=="__main__":
    main()