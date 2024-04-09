from datasets.dairv2x_flow import Dair_v2x
from datasets.dairv2x_trans import Dair_v2x_trans
from datasets.lumpi_flow import LUMPI
from datasets.fedbev_flow import FedbevFlow
from utils import data_util
from datasets.simple_flow import SimpleFlow
from torch.utils.data import DataLoader, ConcatDataset
from datasets.scene_flow import SceneFlow

__all__ = {
    'dair_v2x': Dair_v2x,
    'dair_v2x_val': Dair_v2x,
    'dair_v2x_transform': Dair_v2x,
    'lumpi': LUMPI,
    'ips_train': LUMPI,
    # 'ips_train': Dair_v2x,
    'ips_val': Dair_v2x,
    'dair_v2x_trans': Dair_v2x_trans,
    'fedsimple': SimpleFlow,
    # 'fedbev': FedbevFlow,
    'fedbev_train': LUMPI,
    'fedbev_val': Dair_v2x,
    # 'raw': SimpleFlow
}

__datainfo_unseen__ = {
    'dair_v2x': ['yizhuang02', 'yizhuang09', 'yizhuang10', 'yizhuang13', 'yizhuang16'],
    'dair_v2x_val': ['yizhuang06', 'yizhuang08'],
    'lumpi': ['client0', 'client1', 'client2', 'client3'],
    'ips_train': ['PC1', ],
    'ips_val': ['PC2'],
    'fedbev': ['rsu2', 'rsu3']
}

__datainfo__ = {
    'dair_v2x': ['yizhuang02', 'yizhuang06', 'yizhuang08', 'yizhuang09', 'yizhuang10', 'yizhuang13', 'yizhuang16'],
    'dair_v2x_transform': ['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
    'lumpi': ['client0', 'client1', 'client2', 'client3', 'client4'],
    'dair_v2x_trans': ['yizhuang02', 't1', 'yizhuang06', 't2', 'yizhuang08', 't3', 'yizhuang09', 't4',
                       'yizhuang10', 't5', 'yizhuang13', 't6', 'yizhuang16', 't7'],
    'fedsimple': ['2023-05-26-17-30-12', '2023-03-23-14-39-30', '2023-03-28-17-46-13', '2023-05-09-17-20-24'],
    'ips_train': ['PC1', 'PC2'],
    'ips_val': ['PC1', 'PC2'],
    'fedbev_train': ['rsu1', 'rsu2', 'rsu3'],
    'fedbev_val': ['rsu1', 'rsu2', 'rsu3']
}


def get_dataloaders(configs, args):
    # global datainfo
    dataset_cfg = configs['data']
    train_dls, val_dls, dataset_size_list = [], [], []
    # get train_loaders
    if configs['evaluation_mode'] == 'unseen':
        datainfo = __datainfo_unseen__
        merge = True
    else:
        datainfo = __datainfo__
        merge = False


    if args.alg == 'central':

        for dataset_name, dataset_path in zip(dataset_cfg['train_dataset'], dataset_cfg['train_dataset_path']):
            client_scene = 'all'
            train_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene, split='train', merge=merge)
            train_dls.append(train_dataset)
        print(len(train_dls))
        train_dataset = ConcatDataset(train_dls)
        train_dls = [DataLoader(train_dataset,
                   batch_size=configs['batch_size'],
                   num_workers=configs['num_workers'],
                   pin_memory=True,
                   shuffle=True)]
        # train_dls = [d for dl in train_dls for d in dl]
    else:
        for dataset_name, dataset_path in zip(dataset_cfg['train_dataset'], dataset_cfg['train_dataset_path']):
            for client_scene in datainfo[dataset_name]:
                if configs['scene'] == 'split' and dataset_name != 'dair_v2x':

                    train_dataset1, train_dataset2 = build_dataset2(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene, split='train', merge=merge)
                    train_dls.append(DataLoader(train_dataset1,
                                        batch_size=configs['batch_size'],
                                        num_workers=configs['num_workers'],
                                        pin_memory=True,
                                        shuffle=True))
                    dataset_size_list.append(len(train_dataset1))
                    # train_dls.append(DataLoader(train_dataset2,
                    #                     batch_size=configs['batch_size'],
                    #                     num_workers=configs['num_workers'],
                    #                     pin_memory=True,
                    #                     shuffle=True))
                    #
                    # dataset_size_list.append(len(train_dataset2))
                else:
                    train_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene, split='train', merge=merge)
                    train_dls.append(DataLoader(train_dataset,
                                        batch_size=configs['batch_size'],
                                        num_workers=configs['num_workers'],
                                        pin_memory=True,
                                        shuffle=True))

                    dataset_size_list.append(len(train_dataset))
    # get val_loaders
    if configs['evaluation_mode'] in ('gen-seen', 'unseen', 'ft', 'local'):
        for dataset_name, dataset_path in zip(dataset_cfg['val_dataset'], dataset_cfg['val_dataset_path']):
            for client_scene in datainfo[dataset_name]:
                val_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene,
                                            split='test', merge=False)
                val_dls.append(DataLoader(val_dataset,
                                          batch_size=1,
                                          num_workers=configs['num_workers'],
                                          pin_memory=True,
                                          shuffle=False))
    return train_dls, val_dls, dataset_size_list


def build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene, split='train', merge=False):

    # data augmentation transform
    train_transform = data_util.Augmentation(dataset_cfg['data_augmentation']['aug_together'],
                                              dataset_cfg['data_augmentation']['aug_pc2'],
                                              dataset_cfg['data_process'],
                                              dataset_cfg['num_points'])

    test_transform = data_util.ProcessData(dataset_cfg['data_process'],
                                            dataset_cfg['data_augmentation']['aug_pc2'],
                                            dataset_cfg['num_points'],
                                            dataset_cfg['allow_less_points'])

    dataset = SceneFlow(split=split,
                                    merge=merge,
                                    transform=train_transform,
                                    num_points=dataset_cfg['num_points'],
                                    data_root=dataset_path,
                                    dataset_name=dataset_name,
                                    scene=client_scene,
                                    huafen=0)

    return dataset


# for split one client into two
def build_dataset2(dataset_cfg, dataset_name, dataset_path, client_scene, split='train', merge=False):
    # data augmentation transform
    train_transform = data_util.Augmentation(dataset_cfg['data_augmentation']['aug_together'],
                                             dataset_cfg['data_augmentation']['aug_pc2'],
                                             dataset_cfg['data_process'],
                                             dataset_cfg['num_points'])

    test_transform = data_util.ProcessData(dataset_cfg['data_process'],
                                           dataset_cfg['data_augmentation']['aug_pc2'],
                                           dataset_cfg['num_points'],
                                           dataset_cfg['allow_less_points'])

    dataset1 = __all__[dataset_name](split=split,
                                    merge=merge,
                                    transform=train_transform,
                                    num_points=dataset_cfg['num_points'],
                                    data_root=dataset_path,
                                    dataset_name=dataset_name,
                                    scene=client_scene,
                                    huafen=1)# huafen 1 assign which part of the splited
                                             # 0 or None means dont split
    dataset2 = __all__[dataset_name](split=split,
                                    merge=merge,
                                    transform=train_transform,
                                    num_points=dataset_cfg['num_points'],
                                    data_root=dataset_path,
                                    dataset_name=dataset_name,
                                    scene=client_scene,
                                    huafen=2)

    return dataset1, dataset2