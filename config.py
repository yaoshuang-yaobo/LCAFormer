import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.DATA = edict()
__C.TRAIN = edict()
__C.LOSS = edict()

__C.DATA.BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Datasets root
__C.DATA.IMAGE_PATH = os.path.join(__C.DATA.BASE_PATH, 'data/Luding/')
# tensorboard save
__C.DATA.TENSERBOARD_PATH = os.path.join(__C.DATA.BASE_PATH, 'visualization/tensorboard')
# weight files
__C.DATA.WEIGHTS_PATH = os.path.join(__C.DATA.BASE_PATH, 'weights')
# test results
__C.DATA.PRE_PATH = os.path.join(__C.DATA.BASE_PATH, 'Results/')

__C.TRAIN.WEIGHT = 256
__C.TRAIN.WIDTH = 256
__C.TRAIN.CLASSES = 2
# batch_size
__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.EPOCHS = 400
# thread
__C.TRAIN.WORKERS = 0
# Whether to continue training
__C.TRAIN.RESUME = False
# learning rate
# __C.TRAIN.LR = 1e-4
__C.TRAIN.LR = 0.001800
# MOMENTUM
__C.TRAIN.MOMENTUM = 0.9
# weight-decay   default: 5e-4
__C.TRAIN.WEIGHT_DECAY = 1e-4
__C.TRAIN.WARMUP_FACTOR = 1.0 / 3
__C.TRAIN.WARMUP_ITERS = 0
__C.TRAIN.WARMUP_METHOD = 'linear'
__C.TRAIN.SKIP_VAL = False

__C.TRAIN.VAL_EPOCH = 1
__C.TRAIN.SAVE_EPOCH = 10
__C.TRAIN.LOG_ITER = 10
__C.TRAIN.LOG_DIR = 'log/'


# The weight of the auxiliary loss
__C.LOSS.DC= 0.04

# The weight of the ohem loss
__C.LOSS.USE_OHEM = False
# Auxiliary loss
__C.LOSS.AUX = False
# The weight of the auxiliary loss
__C.LOSS.AUX_WEIGHT = 0.05
