import sys, os, json
import os.path as osp
import numpy as np

from datetime import datetime
from pypcd import pypcd

import torch.utils.data as data


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


class FedbevFlow(data.Dataset):
    # TODO: Reorganize this dataset
    def __init__(self,
                 split,
                 merge,
                 transform,
                 num_points,
                 data_root,
                 dataset_name,
                 scene,
                 huafen=None,
                 time_interval=2,
                 interval_ratio=0.1):
        """
        time_interval: the interval number between two frames. one interval is 0.1s (10Hz)
        interval_ratio: the tolerance threshold of interval
        """
        self.split = split
        assert (split in ['train', 'val', 'test'])
        self.root = data_root
        self.transform = transform
        self.num_points = num_points
        # self.remove_ground = remove_ground
        self.remove_ground = False
        self.merge = merge
        self.time_interval = time_interval
        self.ratio = interval_ratio

        self.dataset_name = dataset_name
        if scene == 'all':
            if self.merge:
                paths = ['rsu2', 'rsu3']
            else:
                paths = ['rsu1', 'rsu2', 'rsu3']
        else:
            paths = [scene]


        self.huafen = huafen
        self.samples = []

        for path in paths:
            tmp_path = osp.join(path, self.split)
            file_list = os.listdir(os.path.join(self.root, tmp_path))
            ordered_list = sorted(range(len(file_list)), key=lambda k: file_list[k])
            file_list = [file_list[i] for i in ordered_list]
            # print(len(file_list))
            for idx, d in enumerate(file_list):

                cur_path = d
                cur_time_index = cur_path[2:6]

                if idx != (len(file_list) -1):
                    next_path = file_list[idx + 1]
                    next_time_index = next_path[2:6]

                    if int(cur_time_index) + 2 != int(next_time_index):
                        # print(cur_time_index, next_time_index)
                        # count+=1
                        continue
                else:
                    break

                pair = {'cur_idx': idx,
                        'next_idx': idx + 1,
                        'cur_pc_path': os.path.join(self.root, tmp_path, cur_path),
                        'next_pc_path': os.path.join(self.root, tmp_path, next_path)}
                self.samples.append(pair)



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        only support self-supervised training w/o gt
        """
        pc1_loaded, pc2_loaded = self.pc_loader_simple(self.samples[index])
        if False:  # if self.split == 'train'
            pc1_transformed, pc2_transformed, _ = self.transform([pc1_loaded, pc2_loaded, None])
        else:
            pc1_transformed, pc2_transformed = pc1_loaded, pc2_loaded  # w/o pc2 aug
        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed

        sample_data = {
            'pc1_transformed': pc1_transformed,
            'pc2_transformed': pc2_transformed,
            'pc1_norm': pc1_norm,
            'pc2_norm': pc2_norm,
            'flow_transformed': np.zeros((self.num_points, 3)),
        }
        return sample_data

    def pc_loader_simple(self, pair):
        pc1_file = pair['cur_pc_path']
        pc2_file = pair['next_pc_path']

        # pc1_data = read_pcd(pc1_file)
        # pc2_data = read_pcd(pc2_file)
        pc1_data = np.load(pc1_file)
        pc2_data = np.load(pc2_file)

        # print(pc1_data.shape)
        # print(pc2_data.shape)

        n1 = len(pc1_data)
        # print(n1)
        n2 = len(pc2_data)
        # print(n2)
        full_mask1 = np.arange(n1)
        full_mask2 = np.arange(n2)

        if n1 >= self.num_points:
            sample_idx1 = np.random.choice(full_mask1, self.num_points, replace=False)
        elif n1 > 0:
            sample_idx1 = np.concatenate(
                (np.arange(n1), np.random.choice(full_mask1, self.num_points - n1, replace=True)), axis=0)
        else:
            sample_idx1 = np.zeros(1, dtype=np.int)
        if n1 != 0:
            pc1_ = pc1_data[sample_idx1, :]
        else:
            pc1_ = np.zeros((self.num_points, 3), dtype=np.int)

        pc1 = pc1_.astype('float32')

        if n2 >= self.num_points:
            sample_idx2 = np.random.choice(full_mask2, self.num_points, replace=False)
        elif n2 > 0:
            sample_idx2 = np.concatenate(
                (np.arange(n2), np.random.choice(full_mask2, self.num_points - n2, replace=True)), axis=0)
        else:
            sample_idx2 = np.zeros(1, dtype=np.int)
        # pc2_ = pc2_data[sample_idx2, :]
        if n2 != 0:
            pc2_ = pc2_data[sample_idx2, :]
        else:
            pc2_ = np.zeros((self.num_points, 3), dtype=np.int)

        pc2 = pc2_.astype('float32')

        pc1 = pc1[:, :3]
        pc2 = pc2[:, :3]

        # print(pc1.shape)
        # print(pc2.shape)

        return pc1, pc2
