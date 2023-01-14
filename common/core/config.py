from easydict import EasyDict as edict
class BaseConfig():

    def __init__(self) -> None:
        
        self.DEVICE = edict()
        self.DEVICE._GPUS = "0"      # available gpus
        self.DEVICE.WHICH = "cuda:0" # which gpu used

        self.HPARAM = edict()
        self.HPARAM.RAND_SEED = 3407 # torch.manual_seed(3407) is all you need
        self.HPARAM.MAX_EPOCHS= 100
        self.HPARAM.BATCH_SIZE = 128
        self.HPARAM.NUM_WORKERS = 0
        
        self.HPARAM.WD = 1e-3 # weight decay
        self.HPARAM.LR = 1e-3 # learning rate
        self.HPARAM.LRD = 0.97 # learning rate decay
        self.HPARAM.CLIP = 3 # gradient clipping

        self.HPARAM.LR_STEP = [70, 90, 110]
        self.HPARAM.LR_FACTOR = 0.1

        self.HPARAM.BN_MOMENTUM = 0.1

        self.DATA = edict()

        self.MODEL = edict()
