"""
PointPWC-Net model and losses
Author: Wenxuan Wu
Date: May 2020
"""

import torch.nn as nn
import torch
from torch.nn import Module
import numpy as np
import torch.nn.functional as F
# from models.pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, PointConvFlow
# from models.pointconv_util import SceneFlowEstimatorPointConv
# from models.pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
from losses.pointconv_util import index_points_group, square_distance

import time


def curvature(pc):
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow

def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows):
    f_curvature = 0.3
    # f_curvature = 0.0
    f_smoothness = 1.0
    f_chamfer = 1.0

    #num of scale
    num_scale = len(pred_flows)

    alpha = [0.02, 0.04, 0.08, 0.16]
    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        cur_pc1 = pc1[i] # B 3 N
        cur_pc2 = pc2[i]
        cur_flow = pred_flows[i] # B 3 N

        #compute curvature
        # print("cur_pc2", cur_pc2.shape)
        cur_pc2_curvature = curvature(cur_pc2)
        # print("cur_pc2", cur_pc2.shape)
        # print("cur_pc2_curvature", cur_pc2_curvature.shape)

        cur_pc1_warp = cur_pc1 + cur_flow
        dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
        moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

        chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()

        #smoothness
        smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

        #curvature
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()

        chamfer_loss += alpha[i] * chamferLoss
        smoothness_loss += alpha[i] * smoothnessLoss
        curvature_loss += alpha[i] * curvatureLoss

    total_loss = f_chamfer * chamfer_loss + f_curvature * curvature_loss + f_smoothness * smoothness_loss

    return total_loss, chamfer_loss, curvature_loss, smoothness_loss


class PointPWCMultiScaleLoss(Module):
    def __init__(self, chamfer_w=1.0, smooth_w=1.0, curvature_w=0.3, alpha=[0.02, 0.04, 0.08, 0.16], reduction='sum', **kwargs):
        super(PointPWCMultiScaleLoss, self).__init__()
        self.chamfer_w = chamfer_w
        self.smooth_w = smooth_w
        self.curvature_w = curvature_w
        self.alpha = alpha

        self.reduction = reduction
        
    # def forward(self, pc1, pc2, pred_flows):
    def forward(self, batch_data, configs):
        pc1 = batch_data['output']['pc1']
        pc2 = batch_data['output']['pc2']
        pred_flows = batch_data['output']['flows']
        num_scale = len(pred_flows)

        chamfer_loss = torch.zeros(1).cuda()
        smoothness_loss = torch.zeros(1).cuda()
        curvature_loss = torch.zeros(1).cuda()

        for i in range(num_scale):
            cur_pc1 = pc1[i] # B 3 N
            cur_pc2 = pc2[i]
            cur_flow = pred_flows[i] # B 3 N

            #compute curvature
            # print("cur_pc2", cur_pc2.shape)
            cur_pc2_curvature = curvature(cur_pc2)
            # print("cur_pc2", cur_pc2.shape)
            # print("cur_pc2_curvature", cur_pc2_curvature.shape)

            cur_pc1_warp = cur_pc1 + cur_flow
            dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
            moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

            if self.reduction == 'sum':
                chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()
            elif self.reduction == 'mean':
                chamferLoss = dist1.mean(dim = 1).mean() + dist2.mean(dim = 1).mean()

            #smoothness
            if self.reduction == 'sum':
                smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()
            elif self.reduction == 'mean':
                smoothnessLoss = computeSmooth(cur_pc1, cur_flow).mean(dim = 1).mean()

            #curvature
            inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
            if self.reduction == 'sum':
                curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()
            elif self.reduction == 'mean':
                curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).mean(dim = 1).mean()

            chamfer_loss += self.alpha[i] * chamferLoss
            smoothness_loss += self.alpha[i] * smoothnessLoss
            curvature_loss += self.alpha[i] * curvatureLoss

        total_loss = self.chamfer_w * chamfer_loss + self.curvature_w * curvature_loss + self.smooth_w * smoothness_loss
        # return total_loss, chamfer_loss, curvature_loss, smoothness_loss
        return total_loss



if __name__ == "__main__":

    pass
