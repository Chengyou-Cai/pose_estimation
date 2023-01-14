from common.core.config import BaseConfig

class Config(BaseConfig):

    def __init__(self) -> None:
        super(Config,self).__init__()

        self.DATA.ROOT = r"_data\mpii"
        self.DATA.BGR2RGB = True

        self.DATA.BASE_EXT = 0.
        
        self.DATA.RAND_EXT = False

        self.DATA.TRAIN_X_EXT = 0.25
        self.DATA.TRAIN_Y_EXT = 0.25
        
        self.DATA.EVAL_X_EXT = 0.25
        self.DATA.EVAL_Y_EXT = 0.25

        self.DATA.AUGM_SCALE = True
        self.DATA.PROB_SCALE = 0.5
        self.DATA.SCALE_FACTOR = 0.25 

        self.DATA.AUGM_ROTAT = True
        self.DATA.PROB_ROTAT = 0.5
        self.DATA.ROTAT_FACTOR = 60

        self.DATA.AUGM_FLIP = True
        self.DATA.PROB_FLIP = 0.5

        self.DATA.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]

        self.MODEL.INPUT_SHAPE = (256,256) # height, width
        self.MODEL.OUTPUT_SHAPE = (64,64)
        self.MODEL.ASPECT_RATIO = self.MODEL.INPUT_SHAPE[1]/self.MODEL.INPUT_SHAPE[0] # w/h

