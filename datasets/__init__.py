from utils import data_util
from datasets.scene_flow import SceneFlow
import torch
from torch.utils.data import DataLoader, ConcatDataset


__datainfo__ = {
    'dair_v2x': ['yizhuang02', 'yizhuang06', 'yizhuang08', 'yizhuang09', 'yizhuang10', 'yizhuang13', 'yizhuang16'],
    'lumpi': ['client0', 'client1', 'client2', 'client3', 'client4'],
    'ips300': ['PC1', 'PC2'],
    'campus': ['rsu1', 'rsu2', 'rsu3'],
    'vehicle': ['yizhuang06', 'yizhuang09', 'yizhuang10', 'yizhuang16']
}


def get_dataloaders(configs, args):
    dataset_cfg = configs['data']
    train_dls, val_dls, dataset_size_list = [], [], []
    # get train_loaders
    if args.alg == 'central':
        # for central dataset
        for dataset_name, dataset_path in zip(dataset_cfg['train_dataset'], dataset_cfg['train_dataset_path']):
            client_scene = 'all'
            if configs['scene'] == 'split':
                train_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene,
                                              split='train_cam', merge=False)
            else:
                train_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene, split='train', merge=False)
            train_dls.append(train_dataset)
        print(len(train_dls))
        print([len(x) for x in train_dls])
        train_dataset = ConcatDataset(train_dls)
        if args.ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_dls = [DataLoader(train_dataset,
                                    batch_size=configs['batch_size'],
                                    num_workers=configs['num_workers'],
                                    pin_memory=True,
                                    sampler=sampler)]
        else:
            train_dls = [DataLoader(train_dataset,
                    batch_size=configs['batch_size'],
                    num_workers=configs['num_workers'],
                    pin_memory=True,
                    shuffle=True)]
            # train_dls = [d for dl in train_dls for d in dl]
    else:
        # for fedavg datasets
        p_count = 0
        for dataset_name, dataset_path in zip(dataset_cfg['train_dataset'], dataset_cfg['train_dataset_path']):

            for client_scene in __datainfo__[dataset_name]:
                if configs['scene'] == 'split':
                    # for optical train_cam
                    if p_count < args.p:
                        print(client_scene, p_count)
                        train_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene, split='train_cam', merge=False)
                        p_count += 1
                    else:
                        train_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path,
                                                      client_scene=client_scene, split='train', merge=False)

                else:
                    train_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene, split='train', merge=False)

                if args.ddp:
                    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                    train_dls.append(DataLoader(train_dataset,
                                    batch_size=configs['batch_size'],
                                    num_workers=configs['num_workers'],
                                    pin_memory=True,
                                    sampler=sampler))
                else:
                    train_dls.append(DataLoader(train_dataset,
                                        batch_size=configs['batch_size'],
                                        num_workers=configs['num_workers'],
                                        pin_memory=True,
                                        shuffle=True))

                dataset_size_list.append(len(train_dataset))

    # get val_loaders
    if configs['evaluation_mode'] in ('gen-seen', 'per', 'local'):
        for dataset_name, dataset_path in zip(dataset_cfg['val_dataset'], dataset_cfg['val_dataset_path']):
            for client_scene in __datainfo__[dataset_name]:

                val_dataset = build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene=client_scene,
                                                split='test', merge=False)
                if args.ddp:
                    sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
                    val_dls.append(DataLoader(val_dataset,
                                          batch_size=1,
                                          num_workers=configs['num_workers'],
                                          pin_memory=True,
                                          sampler=sampler))
                else:
                    val_dls.append(DataLoader(val_dataset,
                                            batch_size=1,
                                            num_workers=configs['num_workers'],
                                            pin_memory=True,
                                            shuffle=False))

    return train_dls, val_dls, dataset_size_list


def build_dataset(dataset_cfg, dataset_name, dataset_path, client_scene, split, merge=False):

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
                                    scene=client_scene)

    return dataset


# for split one client into two
# only used for "huafen" (deprecated)
def build_dataset_split(dataset_cfg, dataset_name, dataset_path, client_scene, split='train', merge=False):
    # data augmentation transform
    train_transform = data_util.Augmentation(dataset_cfg['data_augmentation']['aug_together'],
                                             dataset_cfg['data_augmentation']['aug_pc2'],
                                             dataset_cfg['data_process'],
                                             dataset_cfg['num_points'])

    test_transform = data_util.ProcessData(dataset_cfg['data_process'],
                                           dataset_cfg['data_augmentation']['aug_pc2'],
                                           dataset_cfg['num_points'],
                                           dataset_cfg['allow_less_points'])

    dataset1 = SceneFlow(split=split,
                                    merge=merge,
                                    transform=train_transform,
                                    num_points=dataset_cfg['num_points'],
                                    data_root=dataset_path,
                                    dataset_name=dataset_name,
                                    scene=client_scene,
                                    huafen=1)# huafen 1 assign which part of the splited
                                             # 0 or None means dont split
    dataset2 = SceneFlow[dataset_name](split=split,
                                    merge=merge,
                                    transform=train_transform,
                                    num_points=dataset_cfg['num_points'],
                                    data_root=dataset_path,
                                    dataset_name=dataset_name,
                                    scene=client_scene,
                                    huafen=2)

    return dataset1, dataset2