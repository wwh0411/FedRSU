import json
from datetime import datetime
import os
from tqdm import tqdm
import yaml
import os.path as osp
import numpy as np
from collections import Counter
import cv2
from pypcd import pypcd
from matplotlib import pyplot as plt

import sys
sys.path.append('..')
from optical_flow import Model_flow
print(sys.path)
sys.path.append('/GPFS/data/wenhaowang/bev5/FedBEV')
import torch
import numpy as np
import vis_oral
# from data_utils import *


# cam related info:
# lidar_to_next_cam_list: camera parameter
# next_cam_intrinstic_list: camera intrinsic
# cam_mask: pc in image mask
# cam_coord: pc in image coord
# flow: optical flow


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def read_cam_para(vis_id,data_info,root_path):
    value=data_info[vis_id]
    # print(value)
    intrin_file = osp.join(root_path, value['calib_camera_intrinsic_path'])
    cam_K = load_json(intrin_file)["cam_K"]
    cam_intrin = np.array(cam_K).reshape([3, 3], order="C")
    
    # extrinsic
    extrin_file = osp.join(root_path, value['calib_virtuallidar_to_camera_path'])
    extrin_json = load_json(extrin_file)
    l2r_r = np.array(extrin_json["rotation"])
    l2r_t = np.array(extrin_json["translation"])

    return cam_intrin,l2r_r,l2r_t


def lidar2cam(vis_id,pcd,data_info,root_path):
    pc_points = np.array(pcd[:, :3]).T

    cam_intrin,l2r_r,l2r_t=read_cam_para(vis_id,data_info,root_path)

    # transform to cam coordinates
    pc_points = l2r_r @ pc_points + l2r_t
    pc_points_2d = cam_intrin @ pc_points
    pc_points_2d = pc_points_2d.T

    # normalize
    pc_points_2d = pc_points_2d[pc_points_2d[:, 2] > 0]
    pc_points_2d = pc_points_2d[:, :2] / pc_points_2d[:, 2:3]

    pc_points_2d[:,1]=pc_points_2d[:,1]*(1088/1080)

    mask = (pc_points_2d[:, 0] > 0) & (pc_points_2d[:, 0] < 1920) & \
            (pc_points_2d[:, 1] > 0) & (pc_points_2d[:, 1] < 1088)

    #print("mask:",mask.shape)
    #print("pc_points",pc_points_2d.shape)
    
    return mask,pc_points_2d


def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data['x'])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data['y'])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data['z'])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data['intensity'])
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points


def build_idx_to_info(data):
    idx2info = {}
    for elem in data:
        if elem["pointcloud_path"] == "":
            continue
        idx = elem["pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info


def resize_img(img, img_hw):
    '''
    Input size (N*H, W, 3)
    Output size (N*H', W', 3), where (H', W') == self.img_hw
    '''
    img_h, img_w = img.shape[0], img.shape[1]
    img_new = cv2.resize(img, (img_hw[1], img_hw[0]))
    return img_new


def read_img(img_path,img_hw):
    img = cv2.imread(img_path)
    img=resize_img(img,img_hw)
    img = img.transpose(2,0,1)    
    img = img / 255.0
    return torch.from_numpy(img).float().unsqueeze(0)


def save_cam_flow(data_dir,data_info,pc_key,next_pc_key,model):
    """
    Generate optical flow between two frames 
    """

    img1_path = osp.join(data_dir, 'image/'+pc_key+'.jpg')
    img2_path = osp.join(data_dir, 'image/'+next_pc_key+'.jpg')
    img_hw=(1088,1920)
    img1=read_img(img1_path,img_hw)
    img2=read_img(img2_path,img_hw)

    img1, img2 = img1.cuda(), img2.cuda()

    # print("img1:",img1.shape)  # 1,3,1088,1920
    flow = model.inference_flow(img1, img2)
    flow_np = flow[0].detach().cpu().numpy().transpose([1,2,0])  # flow_np:[h,w,2]
    # print(flow_np.shape)  # 1088,1920,2

    return flow_np


def preprocess_optical_flow():

    gpu = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    process_data_path = '/GPFS/data/zuhongliu/dairv2x_new'
    client_names = ["yizhuang02", "yizhuang06", "yizhuang08", "yizhuang09", "yizhuang10", "yizhuang13", "yizhuang16"]
    # client_names = ["yizhuang02"]
    data_splits = ['train']  # only train split need optical info

    origin_data_path = '/GPFS/public/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/infrastructure-side'
    img_data_path = '/GPFS/public/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/infrastructure-side/image'
    # save_data_path = '/GPFS/public/bsf/dair_v2x_optical'
    save_data_path = '/GPFS/data/wenhaowang/bev5/FedBEV'

    inf_idx2info = build_idx_to_info(
        load_json(osp.join(origin_data_path, "data_info.json"))
    )

    # load optical flow model
    model=Model_flow()
    model=model.cuda()

    # model_path = '/GPFS/rhome/zuhongliu/Unsupervised_depth_OpticalFlow_egomotion/models/dair_v2x/flow/iter_29999.pth'
    model_path = '/GPFS/public/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/infrastructure-side/last.pth'
    weights = torch.load(model_path)
    model.load_state_dict(weights['model_state_dict'], strict=False)
    model.eval()

    for client in client_names:
        print("=======")
        print(client)

        split = 'train'
        tmp_path  = os.path.join(process_data_path, client, split)
        samples = []
        for d in os.listdir(tmp_path):
            samples.append(d)
        print(len(samples))
        print(samples[0])

        for sample in tqdm(samples):
            id_1, id_2 = sample.split('.')[0].split('_')
            # print(id_1, id_2)
            
            cam_intrin,l2r_r,l2r_t = read_cam_para(id_1, inf_idx2info, origin_data_path)
            lidar_to_next_cam_list = np.concatenate([l2r_r,l2r_t],axis=1)
            
            # print(cam_intrin.shape)  # 3,3
            # print(l2r_r.shape)  # 3,3
            # print(l2r_t.shape)  # 3,1
            # print(lidar_to_next_cam_list.shape)  # 3,4

            filename = os.path.join(tmp_path, sample)
            with open(filename, 'rb') as fp:
                data = np.load(fp, allow_pickle=True)
                data = data.item()
                pc1_data = data['pc1'].astype('float32')
                pc2_data = data['pc2'].astype('float32')
            # print(pc1_data.shape)  # n1, 3
            # print(pc2_data.shape)  # n2, 3

            ref_pc_N=pc1_data.shape[0]
            cam_mask, cam_coords = lidar2cam(id_1,pc1_data[:,:3], inf_idx2info, origin_data_path)

            # print(cam_mask.shape)  # n1,
            # print(cam_coords.shape)  # n1, 2
            # print(np.sum(cam_mask))  # n1

            # get optical flow
            flow_np = save_cam_flow(origin_data_path, inf_idx2info, id_1, id_2, model)
            print(flow_np[0])
            print(flow_np.shape)  # 1088,1920,2

            from PIL import Image
            flow_color = vis_oral.flow_to_color(flow_np, convert_to_bgr=False)
            print(type(flow_color))
            print(flow_color.shape)
            image = Image.fromarray(flow_color)

            # 保存为png格式的图片
            # image.save('color.png', 'PNG')


            # Display the image
            # plt.savefig(flow_color, 'color.png')
            # exit()

            print('pc_n', ref_pc_N)
            optf = np.zeros((ref_pc_N, 3), dtype=np.float32)
            optf[cam_mask==1,:2]=flow_np[cam_coords[cam_mask==1, 1].astype(np.int32), cam_coords[cam_mask==1, 0].astype(np.int32)]
            optf[cam_mask==1,-1]=1
            print(optf[0])
            print(optf.shape)  # n1, 3

            cam_data = {
                "pc1": pc1_data,
                "pc2": pc2_data,
                "lidar_to_next_cam":lidar_to_next_cam_list,
                "next_cam_intrinstic":cam_intrin,
                "cam_mask":cam_mask,
                "cam_coords":cam_coords,
                "flow":optf
            }

            save_file = os.path.join(save_data_path, client, 'train_cam')
            if not os.path.exists(save_file):
                os.makedirs(save_file)

            save_path = os.path.join(save_file, sample)
            np.save(save_path, cam_data)
            # exit()
            # np.save(os.path.join(save_data_path, sample), cam_data)

            # break

        # break


if __name__=='__main__':
    preprocess_optical_flow()