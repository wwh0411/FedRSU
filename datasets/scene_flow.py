import sys, os, json
import os.path as osp
import numpy as np

from datetime import datetime
from pypcd import pypcd

import torch.utils.data as data


class SceneFlow(data.Dataset):
    def __init__(self,
                 split,
                 transform,
                 num_points,
                 data_root,
                 dataset_name,
                 scene='all',
                 huafen=0,
                 merge=False
                 ):
        """
        dataset for dairv2x, lumpi, ips300, (ours)
        merge(deprecated): merge train & val
        huafen(deprecated): split the dataset to two datasets
        """

        self.split = split
        assert (split in ['train', 'val', 'test'])
        self.root = data_root
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = False
        self.merge = merge
        self.dataset_name = dataset_name
        self.huafen = huafen

        if scene == 'all':  # for centralized training
            if self.dataset_name == 'dair_v2x':
                paths = ['yizhuang02', 'yizhuang06', 'yizhuang08', 'yizhuang09', 'yizhuang10', 'yizhuang13',
                         'yizhuang16']
            elif self.dataset_name == 'lumpi':
                paths = ["client0", "client1", "client2", "client3", "client4"]
            elif self.dataset_name == 'ips_train' or 'ips_val':
                paths = ['PC1', 'PC2']
            elif self.dataset_name == 'fedbev_train' or 'fedbev_val':
                paths = ['rsu1', 'rsu2', 'rsu3']
        elif scene == 'some':
            if self.dataset_name == 'dair_v2x':
                paths = ['yizhuang02', 'yizhuang06', 'yizhuang08', 'yizhuang09', 'yizhuang10']
        else:
            paths = [scene]

        self.samples = []
        for path in paths:
            tmp_path = osp.join(path, self.split)
            for d in os.listdir(osp.join(self.root, tmp_path)):
                self.samples.append(osp.join(tmp_path, d))

        self.pc_list = self.get_pc_num()
        # self.flow_list = self.get_flow_num()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]

        if self.split == 'train':
            pc1_loaded, pc2_loaded = self.pc_loader_train(path)
            pc1_transformed, pc2_transformed = pc1_loaded, pc2_loaded
        elif self.split == 'val' or self.split == 'test':
            mask1_flow, mask2_flow, pc1_loaded, pc2_loaded, flow = self.pc_loader_gt(path)
            pc1_transformed, pc2_transformed, sf_transformed = pc1_loaded, pc2_loaded, flow

        # if False:  # self.split == 'train':
        #     pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded, flow])
        # else:
        #     pc1_transformed, pc2_transformed, sf_transformed = pc1_loaded, pc2_loaded, flow

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed

        if self.split == 'train':
            sample_data = {
                'pc1_transformed': pc1_transformed,
                'pc2_transformed': pc2_transformed,
                'pc1_norm': pc1_norm,
                'pc2_norm': pc2_norm,
                'flow_transformed': np.zeros((self.num_points, 3)),
            }
        elif self.split == 'val' or self.split == 'test':
            sample_data = {
                'pc1_transformed': pc1_transformed,
                'pc2_transformed': pc2_transformed,
                'pc1_norm': pc1_norm,
                'pc2_norm': pc2_norm,
                'flow_transformed': sf_transformed,
            }
        return sample_data

    def pc_loader(self, path):
        """
        two way to sample points:
        1. mask1_tracks_flow in data, sample data according to the mask
            val/test: dairv2x/ips300/(ours)
            pc_loader_gt()
        2. random sampling
            train: lumpi/dairv2x/ips300/(ours)
            pc_loader_train()
        """
        if self.split == 'train':
            pc1_loaded, pc2_loaded = self.pc_loader_train(path)
            pc1_transformed, pc2_transformed = pc1_loaded, pc2_loaded
        elif self.split == 'val' or self.split == 'test':
            mask1_flow, mask2_flow, pc1_loaded, pc2_loaded, flow = self.pc_loader_gt(path)
            pc1_transformed, pc2_transformed, sf_transformed = pc1_loaded, pc2_loaded, flow

        # if False:  # self.split == 'train':
        #     pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded, flow])
        # else:
        #     pc1_transformed, pc2_transformed, sf_transformed = pc1_loaded, pc2_loaded, flow

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed

        if self.split == 'train':
            sample_data = {
                'pc1_transformed': pc1_transformed,
                'pc2_transformed': pc2_transformed,
                'pc1_norm': pc1_norm,
                'pc2_norm': pc2_norm,
                'flow_transformed': np.zeros((self.num_points, 3)),
            }
        elif self.split == 'val' or self.split == 'test':
            sample_data = {
                'pc1_transformed': pc1_transformed,
                'pc2_transformed': pc2_transformed,
                'pc1_norm': pc1_norm,
                'pc2_norm': pc2_norm,
                'flow_transformed': sf_transformed,
            }
        return sample_data

    def pc_loader_gt(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
            flow: ndarray (N, 3) np.float32
        """

        # pc_num_list = []
        filename = os.path.join(self.root, path)

        with open(filename, 'rb') as fp:
            data = np.load(fp, allow_pickle=True)
            data = data.item()
            pc1 = data['pc1'].astype('float32')
            pc2 = data['pc2'].astype('float32')
            flow = data['flow'].astype('float32')

            n1 = len(pc1)
            n2 = len(pc2)
            # pc_num_list.append(n1)
            # print("n1 length: ",n1)
            # print("n2 length: ",n2)
            # print("flow length: ", len(flow))
            mask1_tracks_flow = data["mask1_tracks_flow"]
            mask2_tracks_flow = data["mask2_tracks_flow"]

            full_mask1 = np.arange(n1)
            full_mask2 = np.arange(n2)
            mask1_noflow = np.setdiff1d(full_mask1, mask1_tracks_flow, assume_unique=True)
            mask2_noflow = np.setdiff1d(full_mask2, mask2_tracks_flow, assume_unique=True)

            nonrigid_rate = 0.8
            rigid_rate = 0.2

            if n1 >= self.num_points:
                if int(self.num_points * nonrigid_rate) > len(mask1_tracks_flow):
                    num_points1_flow = len(mask1_tracks_flow)
                    num_points1_noflow = self.num_points - num_points1_flow
                else:
                    num_points1_flow = int(self.num_points * nonrigid_rate)
                    num_points1_noflow = int(self.num_points * rigid_rate) + 1

                try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
                except:
                    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
                sample_idx1_flow = np.random.choice(mask1_tracks_flow, num_points1_flow, replace=False)
                sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))
                mask1_flow = np.arange(len(sample_idx1_flow))
                # sample_idx1=np.random.choice(np.arange(n1),self.num_points,replace=False)

            else:
                sample_idx1 = np.concatenate(
                    (np.arange(n1), np.random.choice(mask1_tracks_flow, self.num_points - n1, replace=True)), axis=0)
                mask1_flow = np.concatenate((mask1_tracks_flow, np.arange(n1, self.num_points)))

            # print(sample_idx1)

            pc1_ = pc1[sample_idx1, :]
            flow_ = flow[sample_idx1, :]

            pc1 = pc1_.astype('float32')
            flow = flow_.astype('float32')

            if n2 >= self.num_points:
                if int(self.num_points * nonrigid_rate) > len(mask2_tracks_flow):
                    num_points2_flow = len(mask2_tracks_flow)
                    num_points2_noflow = self.num_points - num_points2_flow
                else:
                    num_points2_flow = int(self.num_points * nonrigid_rate)
                    num_points2_noflow = int(self.num_points * rigid_rate) + 1

                try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                    sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=False)
                except:
                    sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=True)
                sample_idx2_flow = np.random.choice(mask2_tracks_flow, num_points2_flow, replace=True)
                sample_idx2 = np.hstack((sample_idx2_flow, sample_idx2_noflow))
                mask2_flow = np.arange(len(sample_idx2_flow))
            else:
                sample_idx2 = np.concatenate(
                    (np.arange(n2), np.random.choice(mask2_tracks_flow, self.num_points - n2, replace=True)), axis=0)
                mask2_flow = np.concatenate((mask2_tracks_flow, np.arange(n2, self.num_points)))

            pc2_ = pc2[sample_idx2, :]
            pc2 = pc2_.astype('float32')

        return mask1_flow, mask2_flow, pc1, pc2, flow

    def pc_loader_train(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        filename = os.path.join(self.root, path)
        # pc_num_list = []
        with open(filename, 'rb') as fp:
            data = np.load(fp, allow_pickle=True)
            data = data.item()
            pc1_data = data['pc1'].astype('float32')
            pc2_data = data['pc2'].astype('float32')

        n1 = len(pc1_data)
        n2 = len(pc2_data)
        # pc_num_list.append(n1)

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

        return pc1, pc2

    def get_pc_num(self):
        pc_num_list = []
        n = len(self.samples)
        # print(n)
        for i in range(n):
            path = self.samples[i]
            filename = os.path.join(self.root, path)

            with open(filename, 'rb') as fp:
                data = np.load(fp, allow_pickle=True)
                data = data.item()
                pc1 = data['pc1'].astype('float32')
                pc2 = data['pc2'].astype('float32')
                # flow = data['flow'].astype('float32')

                n1 = len(pc1)
                # print(n1)
                n2 = len(pc2)
                pc_num_list.append(n1)
        # print(pc_num_list)
        return pc_num_list

    def get_flow_num(self):
        n = len(self.samples)
        flow_li = []
        for i in range(n):
            path = self.samples[i]
            filename = os.path.join(self.root, path)

            with open(filename, 'rb') as fp:
                data = np.load(fp, allow_pickle=True)
                data = data.item()
                pc1 = data['pc1'].astype('float32')
                pc2 = data['pc2'].astype('float32')
                flow = data['flow'].astype('float32')
                # print(flow.shape)
                # print(flow)
                flow_norm = np.linalg.norm(flow, axis=1)

                # print(len(floww_li))
                flow_dyna_li = [x for x in list(flow_norm) if x > 0]
                # print(len(flow_li_li))
                # # print(floww.shape)
                # n1 = len(pc1)
                # # print(n1)
                # n2 = len(pc2)
                # print(type(flow))
                flow_avg = sum(flow_dyna_li) / len(flow_dyna_li)
                # print(flow_avg)
                flow_li.append(flow_avg)
            # break
        # print(pc_num_list)
        return flow_li

    def get_prop(self):
        n = len(self.samples)
        prop_li = []
        for i in range(n):
            path = self.samples[i]
            filename = os.path.join(self.root, path)

            with open(filename, 'rb') as fp:
                data = np.load(fp, allow_pickle=True)
                data = data.item()
                pc1 = data['pc1'].astype('float32')
                pc2 = data['pc2'].astype('float32')
                flow = data['flow'].astype('float32')
                # print(flow.shape)
                # print(flow)
                floww = np.linalg.norm(flow, axis=1)
                floww_li = list(floww)
                # print(len(floww_li))
                flow_li_li = [x for x in floww_li if x > 0]
                prop = len(flow_li_li) / len(floww_li)
                # print(len(flow_li_li))
                # print(floww.shape)

                # print(flow_avg)
                prop_li.append(prop)
            # break
        # print(pc_num_list)
        return prop_li

    def actual_range(self):
        n = len(self.samples)
        max_list = []
        min_list = []
        prop_li = []
        for i in range(n):
            path = self.samples[i]
            filename = os.path.join(self.root, path)

            with open(filename, 'rb') as fp:
                data = np.load(fp, allow_pickle=True)
                data = data.item()
                pc1 = data['pc1'].astype('float32')
                pc2 = data['pc2'].astype('float32')
                # flow = data['flow'].astype('float32')
                max = np.max(pc1, axis=0)
                min = np.min(pc1, axis=0)
                max_list.append(max)
                min_list.append(min)
        max__ = np.max(np.array(max_list), axis=0)
        min__ = np.min(np.array(min_list), axis=0)
        print(max__, min__)
        return