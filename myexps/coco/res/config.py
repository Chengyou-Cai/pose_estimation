from common.core.config import BaseConfig

class Config(BaseConfig):

    def __init__(self) -> None:
        super(Config,self).__init__()

        self.DATA.ROOT = r"_data\coco"
        self.DATA.BGR2RGB = True

        self.DATA.BASE_EXT = 0.05
        
        self.DATA.RAND_EXT = True

        self.DATA.TRAIN_X_EXT = 0.6
        self.DATA.TRAIN_Y_EXT = 0.8
        
        self.DATA.EVAL_X_EXT = 0.01 * 9.0
        self.DATA.EVAL_Y_EXT = 0.015 * 9.0

        self.DATA.AUGM_SCALE = True
        self.DATA.PROB_SCALE = 0.5
        self.DATA.SCALE_FACTOR = 0.25 

        self.DATA.AUGM_ROTAT = True
        self.DATA.PROB_ROTAT = 0.5
        self.DATA.ROTAT_FACTOR = 30

        self.DATA.AUGM_FLIP = True
        self.DATA.PROB_FLIP = 0.5

        self.DATA.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]

        self.DATA.NUM_JOINTS = 17

        # ---
        
        self.MODEL.INPUT_SHAPE = (256,192) # height, width
        self.MODEL.OUTPUT_SHAPE = (64,48)
        self.MODEL.ASPECT_RATIO = self.MODEL.INPUT_SHAPE[1]/self.MODEL.INPUT_SHAPE[0] # w/h

        self.MODEL.NUM_LAYERS=18
        self.MODEL.PRETRAINED=True
        self.MODEL.NUM_DECONV_LAYERS = 3
        self.MODEL.NUM_DECONV_FILTERS = [256, 256, 256]
        self.MODEL.NUM_DECONV_KERNELS = [4, 4, 4]

        self.MODEL.FINAL_CONV_KERNEL = 1
        self.MODEL.DECONV_WITH_BIAS = False
