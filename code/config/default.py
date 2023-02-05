from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yacs.config import CfgNode as CN


_C = CN()

_C.LOGNAME = 'devign_FFmpeg.log'
_C.MODELDIR = ''
_C.MODELNAME = 'DevignModel'
# ----- LOSS BUILDER -----
_C.LOSS = CN()
_C.LOSS.LOSS_TYPE = "CrossEntropy"

_C.LOSS.CostSensitiveCE = CN()
_C.LOSS.CostSensitiveCE.GAMMA = 1.0


_C.LOSS.ClassBalanceCE = CN()
_C.LOSS.ClassBalanceCE.BETA = 0.9999

_C.LOSS.ClassBalanceFocal = CN()
_C.LOSS.ClassBalanceFocal.BETA = 0.999
_C.LOSS.ClassBalanceFocal.GAMMA = 0.5

_C.LOSS.CrossEntropyLabelSmooth = CN()
_C.LOSS.CrossEntropyLabelSmooth.EPSILON = 0.1


_C.LOSS.CrossEntropyLabelAwareSmooth = CN()
_C.LOSS.CrossEntropyLabelAwareSmooth.SMOOTH_HEAD = 0.4
_C.LOSS.CrossEntropyLabelAwareSmooth.SMOOTH_TAIL = 0.1
_C.LOSS.CrossEntropyLabelAwareSmooth.SHAPE = 'concave'

_C.LOSS.FocalLoss = CN()
_C.LOSS.FocalLoss.GAMMA = 2.0

_C.LOSS.LDAMLoss = CN()
_C.LOSS.LDAMLoss.SCALE = 30.0
_C.LOSS.LDAMLoss.MAX_MARGIN = 0.5

_C.LOSS.CDT = CN()
_C.LOSS.CDT.GAMMA = 0.3

_C.LOSS.SEQL = CN()
_C.LOSS.SEQL.GAMMA = 0.9
_C.LOSS.SEQL.LAMBDA = 0.005

_C.LOSS.InfluenceBalancedLoss = CN()
_C.LOSS.InfluenceBalancedLoss.ALPHA = 1000.

_C.LOSS.DiVEKLD = CN()
_C.LOSS.DiVEKLD.POWER_NORM = False
_C.LOSS.DiVEKLD.POWER = 0.5
_C.LOSS.DiVEKLD.TEMPERATURE = 3.0
_C.LOSS.DiVEKLD.ALPHA = 0.5
_C.LOSS.DiVEKLD.BASELOSS = 'CrossEntropy'



def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()