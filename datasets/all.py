import sys, os
import os.path as osp
import numpy as np

import torch.utils.data as data

__all__ = ['Dair_v2x']


class Dair_v2x(data.Dataset):
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
        # assert train is False
        self.split = split
        assert (split in ['train', 'val', 'test'])
        self.root = data_root
        self.transform = transform
        self.num_points = num_points
        # self.remove_ground = remove_ground
        self.remove_ground = False
        self.merge = merge
        self.dataset_name = dataset_name
        self.huafen = huafen

        if scene == 'all':
            if self.dataset_name == 'dair_v2x':
                if self.merge:
                    paths = ['yizhuang02', 'yizhuang09', 'yizhuang10', 'yizhuang13', 'yizhuang16']
                else:
                    paths = ['yizhuang02', 'yizhuang06', 'yizhuang08', 'yizhuang09', 'yizhuang10', 'yizhuang13',
                         'yizhuang16']
            elif self.dataset_name == 'campus':
                if self.merge:
                    paths = ['rsu2', 'rsu3']
                else:
                    paths = ['rsu1', 'rsu2', 'rsu3']



        else:
            paths = [scene]

        self.samples = []


        if self.merge:
            for path in paths:
                tmp_path = osp.join(path, 'train')
                for d in os.listdir(osp.join(self.root, tmp_path)):
                    self.samples.append(osp.join(tmp_path, d))
                tmp_path = osp.join(path, 'val')
                for d in os.listdir(osp.join(self.root, tmp_path)):
                    self.samples.append(osp.join(tmp_path, d))
        else:
            for path in paths:
                # if self.train:
                #     tmp_path = osp.join(path, 'train')
                # else:
                #     tmp_path = osp.join(path, 'val')
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
                    # self.samples.append
                else:
                    for d in os.listdir(osp.join(self.root, tmp_path)):
                        self.samples.append(osp.join(tmp_path, d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mask1_flow, mask2_flow, pc1_loaded, pc2_loaded, flow = self.pc_loader(self.samples[index])
        # For train data, augment together and pc2
        # For val data, augment pc2 or not, also return sf
        if self.split == 'train':
            pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded, flow])
            # print('a')
        else:
            pc1_transformed, pc2_transformed, sf_transformed = pc1_loaded, pc2_loaded, flow  # w/o pc2 aug
        # pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])

        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        if self.split == 'train':
            sample_data = {
                'pc1_transformed': pc1_transformed,
                'pc2_transformed': pc2_transformed,
                'pc1_norm': pc1_norm,
                'pc2_norm': pc2_norm,
                'flow_transformed': sf_transformed,
            }
        else:
            sample_data = {
                'mask1_flow': mask1_flow,
                'mask2_flow': mask2_flow,
                'pc1_transformed': pc1_transformed,
                'pc2_transformed': pc2_transformed,
                'pc1_norm': pc1_norm,
                'pc2_norm': pc2_norm,
                'flow_transformed': sf_transformed,
            }
        # return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]
        return sample_data  # , self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

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
            pc1 = data['pc1'].astype('float32')
            pc2 = data['pc2'].astype('float32')
            flow = data['flow'].astype('float32')

            n1 = len(pc1)
            n2 = len(pc2)
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
                # if num_points1_noflow < 0:
                #     num_points1_noflow = 0
                try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
                except:
                    # if len(mask1_noflow) == 0:
                    #     print('000000000')
                    #     print(full_mask1.shape)  # 4951
                    #     print(full_mask2.shape)  # 5006
                    #     print(mask1_tracks_flow.shape)  # 4951
                    #     print(mask2_tracks_flow.shape)  # 5006
                    #     print(mask1_noflow.shape)  # 0
                    #     print(mask2_noflow.shape)  # 0

                    #     print(self.num_points)  #
                    #     print(num_points1_flow)
                    #     print(num_points1_noflow)
                    #     print('111111111')
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

                # if num_points2_noflow < 0:
                #     num_points2_noflow = 0
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

        # print("pc1:",pc1.shape)
        # print("pc2:",pc2.shape)
        # print("mask1:",mask1_flow)
        # print("mask2:",mask2_flow)

        return mask1_flow, mask2_flow, pc1, pc2, flow


