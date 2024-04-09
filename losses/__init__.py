from losses.unsupervised_losses import UnSupervisedL1LossSeq
from losses.pointpwc_losses import PointPWCMultiScaleLoss
from losses.unsupervised_losses_optical import UnSupervisedL1LossSeqOptical

__all__ = {
    # 'unsup_l1': UnSupervisedL1Loss,
    'unsup_l1_seq': UnSupervisedL1LossSeq,
    'unsup_optical': UnSupervisedL1LossSeqOptical,
    'pointpwc_unsup': PointPWCMultiScaleLoss,
}


def build_criterion(loss_config):
    criterion = __all__[loss_config['loss_type']](**loss_config)

    return criterion