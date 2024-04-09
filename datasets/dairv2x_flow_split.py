import json
from datetime import datetime
import os
from tqdm import tqdm
import yaml
import os.path as osp
import numpy as np
from torch.utils.data import Dataset


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def build_idx_to_info(data):
    idx2info = {}
    for elem in data:
        if elem["pointcloud_path"] == "":
            continue
        idx = elem["pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info


def id_to_str(id, digits=6):
    result = ""
    for i in range(digits):
        result = str(id % 10) + result
        id //= 10
    return result


def split_infra_data_flow():
    return


class DAIRFlowDataset(Dataset):
    def __init__(self, params, visualize=True):
        self.params = params
        self.visualize = visualize
        
        self.root_dir = params['data_dir']
        if self.train:
            split_dir = params['train_split']
        else:
            split_dir = params['val_split']
        self.data_split = load_json(split_dir)

        self.inf_idx2info = build_idx_to_info(
            load_json(osp.join(self.root_dir, "infrastructure-side/data_info.json"))
        )


    def __getitem__(self, idx):
        return


if __name__=='__main__':
    config_file = '/GPFS/data/shfang/repository/self-supervise-pnp/config/dairv2x_nsfp.yaml'
    with open(config_file) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    print(configs)
    # dataset = DAIRFlowDataset(configs)

    inf_idx2info = build_idx_to_info(
        load_json(osp.join(configs['data']['data_dir'], "infrastructure-side/data_info.json"))
    )

    print(len(inf_idx2info))

    sample_pairs = []
    for key, value in tqdm(inf_idx2info.items()):
        # print(key)
        # print(value)
        next_key = id_to_str(int(key) + 1)
        if next_key in inf_idx2info and \
            (int(next_key) <= int(value['batch_end_id'])):
            timestamp1 = datetime.fromtimestamp(float(value['pointcloud_timestamp']) / 1e6)
            timestamp2 = datetime.fromtimestamp(float(inf_idx2info[next_key]['pointcloud_timestamp']) / 1e6)
            interval = (timestamp2 - timestamp1).total_seconds()
            pair = {'key': key,
                    'next_key': next_key,
                    'batch_id': value['batch_id'],
                    'time_interval': interval}
            sample_pairs.append(pair)

    print(len(sample_pairs))

    batch_num = []
    time_intervals = []
    for pair in sample_pairs:
        if pair['batch_id'] not in batch_num:
            batch_num.append(pair['batch_id'])
        time_intervals.append(pair['time_interval'])
    
    print(len(batch_num))
    print(batch_num)

    time_intervals = np.array(time_intervals)
    print(len(time_intervals))
    valid_intervals = time_intervals[time_intervals<0.15]
    print(len(valid_intervals))

    print('max:', np.max(time_intervals))
    print('min:', np.min(time_intervals))
    print('mean:', np.mean(time_intervals))

    hist, bins = np.histogram(time_intervals, bins=[0,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,1]) 
    print(hist)
    print(bins)

    hist, bins = np.histogram(time_intervals, bins=[0,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,1]) 
    print(hist)
    print(bins)

    from matplotlib import pyplot as plt
    plt.hist(time_intervals, bins=[0,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.4]) 
    plt.title("intervals (s)") 
    plt.show()
    plt.savefig(osp.join('/GPFS/data/shfang/repository/self-supervise-pnp/vis_result/dair_v2x', 'dair_inf_intervals.png'))

    plt.hist(time_intervals, bins=[0,0.08,0.09,0.1,0.11,0.12,0.4]) 
    plt.title("intervals (s)") 
    plt.show()
    plt.savefig(osp.join('/GPFS/data/shfang/repository/self-supervise-pnp/vis_result/dair_v2x', 'dair_inf_intervals_2.png'))