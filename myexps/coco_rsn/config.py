from common.core.config import BaseConfig

class Config(BaseConfig):

    def __init__(self) -> None:
        super(Config,self).__init__()

        self.DATA.ROOT = r"_data\coco"
        self.DATA.BGR2RGB = True

        self.MODEL.INPUT_SHAPE = (256,192) # height, width
        self.MODEL.OUTPUT_SHAPE = (64,48)
        self.MODEL.ASPECT_RATIO = self.MODEL.INPUT_SHAPE[1]/self.MODEL.INPUT_SHAPE[0] # w/h

