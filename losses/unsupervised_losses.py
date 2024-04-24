import torch
from torch.nn import Module, MSELoss, L1Loss
from losses.common_losses import SmoothnessLoss, ChamferLoss


class UnSupervisedL1LossSeq(Module):
    def __init__(self, iters_w, w_data, w_smoothness, smoothness_loss_params, chamfer_loss_params, **kwargs):
        super(UnSupervisedL1LossSeq, self).__init__()
        self.data_loss = ChamferLoss(**chamfer_loss_params)
        self.smoothness_loss = SmoothnessLoss(**smoothness_loss_params)
        self.iters_w = iters_w
        self.w_data = w_data
        self.w_smoothness = w_smoothness

    def forward(self, batch_data, configs):
        pc_source, pc_target, pred_flows = batch_data['pc1_transformed'], batch_data['pc2_transformed'], batch_data['output']['flows']
        assert (len(self.iters_w) == len(pred_flows))
        loss = torch.zeros(1).cuda()
        for i, w in enumerate(self.iters_w):
            w_data = self.w_data[i]
            w_smoothness = self.w_smoothness[i]
            pred_flow = pred_flows[i]
            loss_flow = (w_data * self.data_loss(pc_source, pc_target, pred_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow))
            loss += w * loss_flow

        return loss