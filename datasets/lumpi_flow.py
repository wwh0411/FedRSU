import sys, os, json
import os.path as osp
import numpy as np

from datetime import datetime
from pypcd import pypcd

import torch.utils.data as data


class LUMPI(data.Dataset):
    """
    Args:
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 split,
                 merge,
                 transform,
                 num_points,
                 data_root,
                 dataset_name,
                 scene=None,
                 huafen=None,
                 ):
        
        self.split = split
        assert (split in ['train', 'val', 'test'])
        self.root = data_root
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = False
        self.merge = merge
        self.dataset_name = dataset_name
        self.huafen = huafen

        if scene == 'all':
            if self.merge:
                if self.dataset_name == 'lumpi':
                    paths = ['client0', 'client1', 'client2', 'client3']
                elif self.dataset_name == 'ips_train':
                    paths = ['PC1']
            else:
                if self.dataset_name == 'lumpi':
                    paths = ['client0', 'client1', 'client2', 'client3', 'client4']
                elif self.dataset_name == 'ips_train':
                    paths = ['PC1', 'PC2']
                elif self.dataset_name == 'fedbev_train':
                    paths = ['rsu1', 'rsu2', 'rsu3']
        else:
            paths = [scene]

        self.samples = []
        for path in paths:
            tmp_path = osp.join(path, self.split)
            if self.huafen != 0:
                # samples_list = []
                path_list = os.listdir(osp.join(self.root, tmp_path))

                l = len(path_list)
                path_list1 = path_list[:int(l / 2)]
                path_list2 = path_list[int(l / 2):]
                # print(len(path_list1), len(path_list2))
                if self.huafen == 1:
                    for d in path_list1:
                        self.samples.append(osp.join(tmp_path, d))
                else:
                    for d in path_list2:
                        self.samples.append(osp.join(tmp_path, d))
            else:
                for d in os.listdir(osp.join(self.root, tmp_path)):
                    self.samples.append(osp.join(tmp_path, d))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        only support self-supervised training w/o gt
        """
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        if False:  # if self.split == 'train'
            pc1_transformed, pc2_transformed, _ = self.transform([pc1_loaded, pc2_loaded, None])
        else:
            pc1_transformed, pc2_transformed = pc1_loaded, pc2_loaded # w/o pc2 aug
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
    
    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        filename = os.path.join(self.root, path)

        with open(filename, 'rb') as fp:
            data = np.load(fp, allow_pickle=True)
            data = data.item()
            pc1_data = data['pc1'].astype('float32')
            pc2_data = data['pc2'].astype('float32')

        n1 = len(pc1_data)
        #print(n1)
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
            sample_idx2= np.random.choice(full_mask2, self.num_points, replace=False)
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