import torch
from torch.nn import Module, MSELoss, L1Loss
from losses.common_losses import SmoothnessLoss, ChamferLoss, WeightedChamferLoss


# class UnSupervisedL1Loss(Module):
#     def __init__(self, w_data, w_smoothness, smoothness_loss_params, chamfer_loss_params, **kwargs):
#         super(UnSupervisedL1Loss, self).__init__()
#         self.data_loss = ChamferLoss(**chamfer_loss_params)
#         self.smoothness_loss = SmoothnessLoss(**smoothness_loss_params)
#         self.w_data = w_data
#         self.w_smoothness = w_smoothness

#     def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor, i=0) -> torch.Tensor:
#         if len(self.w_data) == 1:
#             w_data = self.w_data[0]
#             w_smoothness = self.w_smoothness[0]
#         else:
#             w_data = self.w_data[i]
#             w_smoothness = self.w_smoothness[i]
#         loss = (w_data * self.data_loss(pc_source, pc_target, pred_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow))
#         return loss


class UnSupervisedL1LossSeqOptical(Module):
    def __init__(self, iters_w, w_data, w_smoothness, w_consistency, smoothness_loss_params, chamfer_loss_params, **kwargs):
        super(UnSupervisedL1LossSeqOptical, self).__init__()
        self.data_loss = WeightedChamferLoss(**chamfer_loss_params)
        self.smoothness_loss = SmoothnessLoss(**smoothness_loss_params)
        self.iters_w = iters_w
        self.w_data = w_data
        self.w_smoothness = w_smoothness
        self.w_consistency = w_consistency

    def forward(self, batch_data, configs):
        pc_source, pc_target, pred_flows = batch_data['pc1_transformed'], batch_data['pc2_transformed'], batch_data['output']['flows']
        # print('...')
        # print(pc_source.shape)  # torch.Size([2, 4096, 3])
        # print(pc_target.shape)  # torch.Size([2, 4096, 3])
        assert (len(self.iters_w) == len(pred_flows))

        # print(pc_source.shape)  # (b, n, 3)
        # print(pc_target.shape)  # (b, n, 3)

        # print(batch_data['cam_mask'].shape)  # (b, n)
        # print(batch_data['cam_coords'].shape)  # (b, n, 2)
        # print(batch_data['optical_flow'].shape)  # (b, n, 3)
        # print(batch_data['cam_extrin'].shape)  # (b, 3, 4)
        # print(batch_data['cam_intrin'].shape)  # (b, 3, 3)

        cam_mask = batch_data['cam_mask']  # (b, n)
        points_cam_coords = batch_data['cam_coords']  # (b, n, 2)

        lidar_to_future_cam = batch_data['cam_extrin']  # (b, 3, 4)
        future_cam_intrinsic = batch_data['cam_intrin']  # (b, 3, 3)
        opt_flow = batch_data['optical_flow']  # (b, n, 3)



        loss = torch.zeros(1).cuda()
        for i, w in enumerate(self.iters_w):
            w_data = self.w_data[i]
            w_smoothness = self.w_smoothness[i]
            pred_flow = pred_flows[i]

            # === optical related consistency_loss ===
            pc_pred = (pc_source + pred_flow).contiguous()
            # print(pc_pred.shape)  # (b,n,3)

            points_mask = torch.zeros(pc_source.size(0), pc_source.size(1)).to(pc_source.device)
            # print(points_mask.shape)  # b,n

            cond = cam_mask == 1
            # print(cond.shape)  # b,n
            # print(cond.sum())  # b*n

            pred_points_cam = pc_pred  # b,n,3
            select_cam_coords = points_cam_coords  # b,n,2
            selected_flow = opt_flow  # b,n,3

            next_points_cam = torch.matmul(lidar_to_future_cam[:, :, :3], pred_points_cam.permute(0, 2, 1).contiguous()) + \
                            lidar_to_future_cam[:, :, 3].view(-1, 3, 1)  # 3, n
            # print(next_points_cam.shape)  # b,3,n

            eps=1e-8
            points_to_next_cam = torch.matmul(future_cam_intrinsic, next_points_cam)
            # print(points_to_next_cam.shape)  # b,3,n
            points_to_next_cam = points_to_next_cam[:, :2] / (points_to_next_cam[:, 2:] + eps)
            # print(points_to_next_cam.shape)  # b,2,n

            projected_motion = points_to_next_cam.permute(0, 2, 1).contiguous() - select_cam_coords[:, :, :2]  # b,n,2
            # print(projected_motion.shape)  # b,n,2

            # alpha=0.1, tau=5.0
            points_mask = torch.exp(-2.0 * torch.clamp(torch.norm(selected_flow[:, :,:2],dim=2)-0.0, min=0.0)).unsqueeze(1)  #.unsqueeze(0)
            # print(points_mask.shape)  # b,1,n

            # print(selected_flow[:,:,2:].shape)  # b,n,1
            # print(torch.sum(selected_flow[:,:,2:]))  # b*n

            consistency_loss = torch.mean(torch.sum(torch.abs(projected_motion - selected_flow[:,:,:2]) * \
                                    selected_flow[:,:,2:] * points_mask.permute(0,2,1).contiguous(), dim=2))
            # print(consistency_loss)
            # consistency_loss = torch.mean(torch.sum(torch.abs(projected_motion - selected_flow[:,:,:2]), dim=2))
            # print(consistency_loss)
            # ========================================


            # print(pred_flow.shape)  # (b, n, 3)
            cf_loss= self.data_loss(pc_source, pc_target, pred_flow, weights=(1-points_mask).squeeze())
            smoothness_loss = self.smoothness_loss(pc_source, pred_flow)

            loss_flow = w_data * cf_loss + w_smoothness * smoothness_loss + self.w_consistency * consistency_loss

            loss += w * loss_flow

        return loss