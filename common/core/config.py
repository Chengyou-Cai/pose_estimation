from easydict import EasyDict as edict
class BaseConfig():

    def __init__(self) -> None:
        
        self.DEVICE = edict()
        self.DEVICE._gpus = "0"      # available gpus
        self.DEVICE.which = "cuda:0" # which gpu used

        self.HPARAM = edict()
        self.HPARAM.rand_seed = 3407 # torch.manual_seed(3407) is all you need
        self.HPARAM.max_epochs = 100
        self.HPARAM.batch_size = 128
        self.HPARAM.num_workers = 0
        
        self.HPARAM.wd = 1e-3 # weight decay
        self.HPARAM.lr = 1e-3 # learning rate
        self.HPARAM.lrd = 0.97 # learning rate decay
        self.HPARAM.clip = 3 # gradient clipping

        self.DATA = edict()

        self.MODEL = edict()
