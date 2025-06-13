import json
from datetime import datetime
import os
from tqdm import tqdm
import yaml
import os.path as osp
import numpy as np
from collections import Counter
from pypcd import pypcd
from matplotlib import pyplot as plt


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


def preprocess_dair(interval_range = [0.09, 0.11]):
    data_dir = "/GPFS/public/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/infrastructure-side"

    inf_idx2info = build_idx_to_info(
        load_json(osp.join(data_dir, "data_info.json"))
    )

    split_info = load_json(osp.join(data_dir, "split-data.json"))

    print(len(split_info['train']))  # 57
    print(len(split_info['val']))  # 25

    train_pairs = []
    val_pairs = []
    all_pairs = []

    all_locations = []
    train_locations = []
    val_locations = []

    for key, value in inf_idx2info.items():
        next_key = id_to_str(int(key) + 1)
        if next_key not in inf_idx2info or \
            (int(next_key) > int(value['batch_end_id'])):
            continue
        
        timestamp1 = datetime.fromtimestamp(float(value['pointcloud_timestamp']) / 1e6)
        timestamp2 = datetime.fromtimestamp(float(inf_idx2info[next_key]['pointcloud_timestamp']) / 1e6)
        interval = (timestamp2 - timestamp1).total_seconds()
        if interval < interval_range[0] or interval > interval_range[1]:
            continue

        pair = {
            'key': key,
            'next_key': next_key,
            'batch_id': value['batch_id'],
            'time_interval': interval
        }
        if value['batch_id'] in split_info['train']:
            train_pairs.append(pair)
            train_locations.append(value['intersection_loc'])
        elif value['batch_id'] in split_info['val']:
            val_pairs.append(pair)
            val_locations.append(value['intersection_loc'])
        all_pairs.append(pair)
        all_locations.append(value['intersection_loc'])
            
    print(len(train_pairs))  # 6834
    print(len(val_pairs))  # 3152
    print(len(all_pairs))  # 9986

    print(len(all_locations))
    print(len(train_locations))
    print(len(val_locations))
    print("All data:", sorted(dict(Counter(all_locations)).items(), key=lambda kv:kv[0]))
    print("Train data:", sorted(dict(Counter(train_locations)).items(), key=lambda kv:kv[0]))
    print("Val data:", sorted(dict(Counter(val_locations)).items(), key=lambda kv:kv[0]))
    print("All data:", Counter(all_locations))
    print("Train:", Counter(train_locations))
    print("Val:", Counter(val_locations))

    # ===========================
    # # data statistics
    # batch_num = []
    # time_intervals = []
    # for pair in all_pairs:
    #     if pair['batch_id'] not in batch_num:
    #         batch_num.append(pair['batch_id'])
    #     time_intervals.append(pair['time_interval'])

    # print(len(batch_num))
    # time_intervals = np.array(time_intervals)

    # hist, bins = np.histogram(time_intervals, bins=[0,0.09,0.095,0.1,0.105,0.11,1]) 
    # print(hist)  # [   0 1684 4345 3009  948    0]
    # print(bins)
    # ===========================

    # ===========================
    # check lidar coordinate
    locations = list(set(all_locations))
    print(locations)
    s = 0
    for pair in tqdm(all_pairs):
        if inf_idx2info[pair['key']]['intersection_loc'] != 'yizhuang09':
            continue
        virtuallidar_to_world_1 = load_json(os.path.join(data_dir,'calib/virtuallidar_to_world/'+pair['key']+'.json'))
        virtuallidar_to_world_2 = load_json(os.path.join(data_dir,'calib/virtuallidar_to_world/'+pair['next_key']+'.json'))
        # print(virtuallidar_to_world_1)
        # print(virtuallidar_to_world_2)
        rot = virtuallidar_to_world_1['rotation'] == virtuallidar_to_world_2['rotation']
        trans = virtuallidar_to_world_1['translation'] == virtuallidar_to_world_2['translation']
        # print(rot, trans)
        if not (rot and trans):
            print(pair)
            print(virtuallidar_to_world_1['rotation'])
            print(virtuallidar_to_world_2['rotation'])
            s += 1
            # break
        # break
    print(s)
    # ===========================


def crop_lidar():
    data_dir = "/GPFS/public/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/infrastructure-side"

    inf_idx2info = build_idx_to_info(
        load_json(osp.join(data_dir, "data_info.json"))
    )

    print(len(inf_idx2info))

    for key, value in tqdm(inf_idx2info.items()):

        # intrainsic
        intrin_file = osp.join(data_dir, value['calib_camera_intrinsic_path'])
        cam_K = load_json(intrin_file)["cam_K"]
        cam_intrin = np.array(cam_K).reshape([3, 3], order="C")
        # extrinsic
        extrin_file = osp.join(data_dir, value['calib_virtuallidar_to_camera_path'])
        extrin_json = load_json(extrin_file)
        l2r_r = np.array(extrin_json["rotation"])
        l2r_t = np.array(extrin_json["translation"])

        # lidar point cloud
        pcd_path = osp.join(data_dir, value['pointcloud_path'])
        pcd = read_pcd(pcd_path)
        pc_points = np.array(pcd[:, :3]).T

        # transform to cam coordinates
        pc_points = l2r_r @ pc_points + l2r_t
        pc_points_2d = cam_intrin @ pc_points
        pc_points_2d = pc_points_2d.T

        # normalize
        pc_points_2d = pc_points_2d[pc_points_2d[:, 2] > 0]
        pc_points_2d = pc_points_2d[:, :2] / pc_points_2d[:, 2:3]

        # get the mask
        mask = (pc_points_2d[:, 0] > 0) & (pc_points_2d[:, 0] < 1920) & \
                (pc_points_2d[:, 1] > 0) & (pc_points_2d[:, 1] < 1080)
        pcd_mask = pcd[mask]

        # print(pcd_mask.shape)

        # save masked pcd file
        np.save(osp.join(data_dir, 'velodyne_new/{}.npy'.format(key)), pcd_mask)

        # visualize
        # pc_range = [0, -50, -10, 100, 50, 10]
        # canvas = Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
        #                                         canvas_x_range=(pc_range[0], pc_range[3]), 
        #                                         canvas_y_range=(pc_range[1], pc_range[4]),
        #                                         left_hand=False
        #                                     )
        # canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_mask) # Get Canvas Coords
        # canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points

        # plt.axis("off")
        # plt.imshow(canvas.canvas)
        # plt.tight_layout()
        # save_path = osp.join("/GPFS/data/shfang/repository/self-supervise-pnp/vis_result", "test_crop_2.png")
        # plt.savefig(save_path, transparent=False, dpi=400)
        # plt.clf()


if __name__=='__main__':
    interval = [0.09, 0.11]  # the interval between frames
    margin = 0.2 # Add a margin to the cuboids. Some cuboids are very tight and might lose some points.
    max_height = 4  # the max height of point cloud
    max_dist = 50  # the max range of point cloud
    # preprocess_dair(interval_range=interval)

    crop_lidar()