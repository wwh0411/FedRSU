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


class SimpleFlow(data.Dataset):
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

        self.time_interval = time_interval
        self.ratio = interval_ratio

        self.dataset_name = dataset_name
        if scene == 'all':
            paths = ['2023-03-23-14-39-30', '2023-03-28-17-46-13', '2023-05-09-17-20-24']
        else:
            paths = [scene]
        self.data_info_list = []
        for path in paths:
            self.data_info_list.append(osp.join(self.root, path, path + '.json'))

        self.huafen = huafen
        self.initial_data()


    def initial_data(self):
        self.samples = []

        for file in self.data_info_list:
            with open(file, 'r') as json_file:
                self.data_info = json.load(json_file)

            for i in range(len(self.data_info)):
                idx = str(i)
                next_idx = str(i + self.time_interval)
                if idx not in self.data_info or next_idx not in self.data_info:
                    continue

                cur_timestamp = self.data_info[idx]['timestamp']
                next_timestamp = self.data_info[next_idx]['timestamp']
                cur_path = self.data_info[idx]['pointcloud_path']
                next_path = self.data_info[next_idx]['pointcloud_path']
                timestamp1 = datetime.fromtimestamp(float(cur_timestamp.replace('-', '.')))
                timestamp2 = datetime.fromtimestamp(float(next_timestamp.replace('-', '.')))
                interval = (timestamp2 - timestamp1).total_seconds()

                if interval < 0.1 * self.time_interval * (1 - self.ratio) or \
                        interval > 0.1 * self.time_interval * (1 + self.ratio):
                    continue

                pair = {'cur_idx': idx,
                        'next_idx': next_idx,
                        'cur_pc_path': os.path.join(self.root, cur_path),
                        'next_pc_path': os.path.join(self.root, next_path)}
                self.samples.append(pair)
        # huafen
        l = len(self.samples)
        l = int(l / 2)
        if self.huafen == 1:
            self.samples = self.samples[:l]
        else:
            self.samples = self.samples[l:]

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
