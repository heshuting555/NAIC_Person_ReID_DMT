import torch.nn.functional as F
import logging
from .softmax_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .triplet_loss import TripletLoss,WeightedTripletLoss


def make_loss(cfg, num_classes):    # modified by gu
    feat_dim = 2048
    logger = logging.getLogger("reid_baseline.train")
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.WEIGHTED_TRIPLET:
            if cfg.MODEL.NO_MARGIN:
                triplet = WeightedTripletLoss()
                logger.info("using soft margin Weighted Triplet Loss for training")
            else:
                triplet = WeightedTripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
                logger.info("using Weighted Triplet Loss for training with margin:{}".format(cfg.SOLVER.MARGIN))
        else:
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletLoss()
                logger.info("using soft margin triplet loss for training")
            else:
                triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
                logger.info("using Triplet Loss for training with margin:{}".format(cfg.SOLVER.MARGIN))

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)
        logger.info("label smooth on, numclasses:{}".format(num_classes))
    else:
        id_loss_func = F.cross_entropy

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0]
            else:
                return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0]
        else:
            print('unexpected loss type')

    return loss_func, center_criterion