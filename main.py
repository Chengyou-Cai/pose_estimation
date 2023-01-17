from engine.data_drive import DataDrive
from engine.mods_proxy import ModsProxy
from engine.sys_engine import SysEngine, set_environ
from myexps.coco.res.config import Config as COCOResConfig

from datasets.coco_dataset import COCODataset
from models.pose_resnet import PoseResNet

def main():

    config = COCOResConfig()
    set_environ(config=config)

    dat = DataDrive(config,_drive=COCODataset)
    mod = ModsProxy(config,_proxy=PoseResNet)
    sys = SysEngine(config,mods_proxy=mod,data_drive=dat)
    sys.fit()


if __name__=="__main__":
    main()