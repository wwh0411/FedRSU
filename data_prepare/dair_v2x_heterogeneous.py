import sys, os
import os.path as osp
import numpy as np
from tqdm import tqdm
import math
import torch.utils.data as data

from matplotlib import pyplot as plt
from canvas_bev import Canvas_BEV_heading_right


def visualize_pc_and_flow(pc1, pc2, flow, save_path, file_name, pc_range=[0, -32, -10, 128, 32, 10]):
    canvas = Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*40, (pc_range[3]-pc_range[0])*40),
                                                canvas_x_range=(pc_range[0], pc_range[3]), 
                                                canvas_y_range=(pc_range[1], pc_range[4]),
                                                left_hand=False
                                            )
    
    # pc1
    pc1_canvas_xy, pc1_valid_mask = canvas.get_canvas_coords(pc1)
    # pc2
    pc2_canvas_xy, pc2_valid_mask = canvas.get_canvas_coords(pc2)
    flow_xy = canvas.get_canvas_flows(flow)

    # fig pc1+pc2
    canvas.clear_canvas()
    canvas.draw_canvas_points(pc1_canvas_xy[pc1_valid_mask], radius=2, colors=(0,255,0))  # green
    canvas.draw_canvas_points(pc2_canvas_xy[pc2_valid_mask], radius=2, colors=(255,0,0)) 
    plt.axis("off")
    plt.imshow(canvas.canvas)
    plt.tight_layout()
    # save_path = osp.join(save_dir, "{}_{}_{}_{}_{}.png".format(scene, data_split, i, model_epoch, 'gt'))
    save_file = osp.join(save_path, file_name+'_pc.png')
    plt.savefig(save_file, transparent=False, dpi=400)
    plt.clf()

    # fig pc1+flow
    canvas.clear_canvas()
    canvas.draw_canvas_points(pc1_canvas_xy[pc1_valid_mask], radius=2, colors=(0,255,0))  # green
    canvas.draw_canvas_flows(pc1_canvas_xy[pc1_valid_mask], flow_xy[pc1_valid_mask], colors=(255,0,0))
    plt.axis("off")
    plt.imshow(canvas.canvas)
    plt.tight_layout()
    # save_path = osp.join(save_dir, "{}_{}_{}_{}_{}.png".format(scene, data_split, i, model_epoch, 'gt'))
    save_file = osp.join(save_path, file_name+'_flow.png')
    plt.savefig(save_file, transparent=False, dpi=400)
    plt.clf()

    return


def rotate_data(pc1_, pc2_, flow_, angle_degree):
    pc1 = np.copy(pc1_)
    pc2 = np.copy(pc2_)
    flow = np.copy(flow_)
    angle = np.deg2rad(angle_degree)
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rot_matrix = np.array([[cosval, sinval, 0],
                            [-sinval, cosval, 0],
                            [0, 0, 1]], dtype=np.float32)
    pc1_transformed = pc1 + flow

    pc1[:, :3] = pc1[:, :3].dot(rot_matrix)
    pc1_transformed[:, :3] = pc1_transformed[:, :3].dot(rot_matrix)
    pc2[:, :3] = pc2[:, :3].dot(rot_matrix)
    flow = pc1_transformed - pc1
    return pc1, pc2, flow


def flip_data(pc1_, pc2_, flow_, axis):
    pc1 = np.copy(pc1_)
    pc2 = np.copy(pc2_)
    flow = np.copy(flow_)
    pc1[:, axis:axis+1] = -pc1[:, axis:axis+1]
    pc2[:, axis:axis+1] = -pc2[:, axis:axis+1]
    flow[:, axis:axis+1] = -flow[:, axis:axis+1]
    return pc1, pc2, flow


def rescale_data(pc1_, pc2_, flow_, scale):
    pc1 = np.copy(pc1_)
    pc2 = np.copy(pc2_)
    flow = np.copy(flow_)
    scale = np.diag([scale, scale, scale]).astype(np.float32)
    pc1_transformed = pc1 + flow
    pc1[:, :3] = pc1[:, :3].dot(scale)
    pc1_transformed[:, :3] = pc1_transformed[:, :3].dot(scale)
    pc2[:, :3] = pc2[:, :3].dot(scale)
    flow = pc1_transformed - pc1
    return pc1, pc2, flow


def downsample_data(pc1_, pc2_, mask1_tracks_flow_, mask2_tracks_flow_, flow_, rate):
    pc1 = np.copy(pc1_)
    pc2 = np.copy(pc2_)
    mask1_tracks_flow = np.copy(mask1_tracks_flow_)
    mask2_tracks_flow = np.copy(mask2_tracks_flow_)
    flow = np.copy(flow_)
    n1 = len(pc1)
    n2 = len(pc2)
    full_mask1 = np.arange(n1)
    full_mask2 = np.arange(n2)
    mask1_noflow = np.setdiff1d(full_mask1, mask1_tracks_flow, assume_unique=True)
    mask2_noflow = np.setdiff1d(full_mask2, mask2_tracks_flow, assume_unique=True)
    print(mask1_tracks_flow.shape)
    print(mask1_noflow.shape)

    pc1_sample_flow = np.random.choice(mask1_tracks_flow,
                                       int(len(mask1_tracks_flow) * rate),
                                       replace=False)
    pc1_sample_noflow = np.random.choice(mask1_noflow,
                                       int(len(mask1_noflow) * rate),
                                       replace=False)

    print(pc1_sample_flow.shape)
    print(pc1_sample_noflow.shape)
    pc1_sample = np.hstack((pc1_sample_flow, pc1_sample_noflow))
    print(pc1_sample.shape)
    mask1_flow = np.arange(len(pc1_sample_flow))
    print(mask1_flow)
    pc1 = pc1[pc1_sample, :].astype('float32')
    flow = flow[pc1_sample, :].astype('float32')

    pc2_sample_flow = np.random.choice(mask2_tracks_flow, 
                                       int(len(mask2_tracks_flow) * rate),
                                       replace=False)
    pc2_sample_noflow = np.random.choice(mask2_noflow, 
                                       int(len(mask2_noflow) * rate),
                                       replace=False)
    pc2_sample = np.hstack((pc2_sample_flow, pc2_sample_noflow))
    mask2_flow = np.arange(len(pc2_sample_flow))

    pc2 = pc2[pc2_sample, :].astype('float32')
    print(pc1.shape)
    return pc1, pc2, mask1_flow, mask2_flow, flow


def crop_data(pc1_, pc2_, flow_, crop_rate_x, crop_rata_y, crop_rate_z):
    pc1 = np.copy(pc1_)
    pc2 = np.copy(pc2_)
    flow = np.copy(flow_)


    pc1_max = np.max(np.abs(pc1), axis=0)
    new1 = []
    new3 = []
    for i, (x, y, z) in enumerate(np.abs(pc1)):
        if x < pc1_max[0] * crop_rate_x:
            if y < pc1_max[1] * crop_rata_y:
                if z < pc1_max[2] * crop_rate_z:
                    new1.append(pc1[i])
                    new3.append(flow[i])
    pc1 = np.array(new1)
    flow = np.array(new3)

    pc2_max = np.max(np.abs(pc2), axis=0)
    new2 = []
    for i, (x, y, z) in enumerate(pc2):
        if x < pc2_max[0] * crop_rate_x:
            if y < pc2_max[1] * crop_rata_y:
                if z < pc2_max[2] * crop_rate_z:
                    new2.append(pc2[i])
    pc2 = np.array(new2)

    mask1 = np.arange(pc1.shape[0])
    mask2 = np.arange(pc2.shape[0])
    return pc1, pc2, mask1, mask2, flow


def fov_data(pc1_, pc2_, flow_, angle_new):
    pc1 = np.copy(pc1_)
    pc2 = np.copy(pc2_)
    flow = np.copy(flow_)

    rad = math.radians(angle_new)
    count = 0

    new1 = []
    new3 = []
    l1 = np.mean(pc1, axis=0)[1]
    for i, (x, y, z) in enumerate(pc1):
        if math.fabs(y - l1) / 2 > math.fabs(x * math.sin(rad)):
            count+=1
        else:
            new1.append([x, y, z])
            new3.append(flow[i])
    pc1 = np.array(new1)
    flow = np.array(new3)
    print(count)

    new2 = []
    for x, y, z in pc2:
        if math.fabs(y - l1) / 2 > math.fabs(x * math.sin(rad)):
            count += 1
        else:
            new2.append([x, y, z])
    pc2 = np.array(new2)
    print(count)

    print(count)

    mask1 = np.arange(pc1.shape[0])
    mask2 = np.arange(pc2.shape[0])
    return pc1, pc2, mask1, mask2, flow


def main():
    source_dir = '/GPFS/data/zuhongliu/dair_v2x_posttracking/yizhuang10/'
    target_dir = '/GPFS/public/bsf/dair_v2x_transform1/'
    target_dir = '/GPFS/data/wenhaowang/dataset/dair_v2x_transform/'
    #os.makedirs(target_dir + 't10/val')
    # transform_pattern:
    # t1: rotate 30
    # t2: rotate 60
    # t3: flip x
    # t4: flip y
    # t5: flip z
    # t6: rescale 0.8
    # t7: rescale 1.2
    # t8: downsample 0.8
    # t9: downsample 0.6

    transform_list = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10']
    split = 'val' # train / val
    for t in transform_list:
        if not osp.exists(osp.join(target_dir, t)):
            os.mkdir(osp.join(target_dir, t))
        if not osp.exists(osp.join(target_dir, t, split)):
            os.mkdir(osp.join(target_dir, t, split))
    
    # train
    source_dir = osp.join(source_dir, split)
    all_files = []
    all_filenames = []
    for d in os.listdir(source_dir):
        all_files.append(osp.join(source_dir, d))
        all_filenames.append(d)
    
    print(len(all_files))

    for i, filepath in tqdm(enumerate(all_files)):
        print(filepath)
        with open(filepath, 'rb') as fp:
            data = np.load(fp, allow_pickle=True)
            data = data.item()
        # print(data.keys())
        pc1 = data['pc1'].astype('float32')
        pc2 = data['pc2'].astype('float32')
        #print(pc2)
        #print(pc1.shape)
        #print(pc2.shape)

        flow = data['flow'].astype('float32')
        #print(flow)
        #print(flow.shape)
        mask1_tracks_flow = data["mask1_tracks_flow"]
        mask2_tracks_flow = data["mask2_tracks_flow"]

        # shift pc location
        new_range = [-60, -40, -10, 60, 40, 10]
        shift = np.mean(pc1, axis=0)

        pc1[:, :3] = pc1[:, :3] - shift # 感觉这里代码有点累赘
        pc2[:, :3] = pc2[:, :3] - shift

        #print(pc2)
        save_dir = "/GPFS/data/wenhaowang/FedBEV/"



        # # t1: rotate 30
        # pc1_new, pc2_new, flow_new = rotate_data(pc1, pc2, flow, angle_degree=30)
        # data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask1_tracks_flow, 'mask2_tracks_flow':mask2_tracks_flow}
        # np.save(osp.join(target_dir, 't1', split, all_filenames[i]), data)



        # # t2: rotate 60
        # pc1_new, pc2_new, flow_new = rotate_data(pc1, pc2, flow, angle_degree=60)
        # data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask1_tracks_flow, 'mask2_tracks_flow':mask2_tracks_flow}
        # np.save(osp.join(target_dir, 't2', split, all_filenames[i]), data)

        # # t3: flip x
        # pc1_new, pc2_new, flow_new = flip_data(pc1, pc2, flow, axis=0)
        # data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask1_tracks_flow, 'mask2_tracks_flow':mask2_tracks_flow}
        # np.save(osp.join(target_dir, 't3', split, all_filenames[i]), data)

        # # t4: flip y
        # pc1_new, pc2_new, flow_new = flip_data(pc1, pc2, flow, axis=1)
        # data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask1_tracks_flow, 'mask2_tracks_flow':mask2_tracks_flow}
        # np.save(osp.join(target_dir, 't4', split, all_filenames[i]), data)

        # # t5: flip z
        # pc1_new, pc2_new, flow_new = flip_data(pc1, pc2, flow, axis=2)
        # data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask1_tracks_flow, 'mask2_tracks_flow':mask2_tracks_flow}
        # np.save(osp.join(target_dir, 't5', split, all_filenames[i]), data)

        # # t6: rescale 0.8
        # pc1_new, pc2_new, flow_new = rescale_data(pc1, pc2, flow, scale=0.8)
        # data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask1_tracks_flow, 'mask2_tracks_flow':mask2_tracks_flow}
        # np.save(osp.join(target_dir, 't6', split, all_filenames[i]), data)

        # # t7: rescale 1.2
        # pc1_new, pc2_new, flow_new = rescale_data(pc1, pc2, flow, scale=1.2)
        # data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask1_tracks_flow, 'mask2_tracks_flow':mask2_tracks_flow}
        # np.save(osp.join(target_dir, 't7', split, all_filenames[i]), data)

        # # t8: downsample 0.8
        # pc1_new, pc2_new, mask_pc1_new, mask_pc2_new, flow_new = downsample_data(pc1, pc2, mask1_tracks_flow, mask2_tracks_flow, flow, rate=0.8)
        # data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask_pc1_new, 'mask2_tracks_flow':mask_pc2_new}
        # np.save(osp.join(target_dir, 't8', split, all_filenames[i]), data)

        # t9: downsample 0.6
        #pc1_new, pc2_new, mask_pc1_new, mask_pc2_new, flow_new = downsample_data(pc1, pc2, mask1_tracks_flow, mask2_tracks_flow, flow, rate=0.6)
        #data = {'pc1':pc1_new, 'pc2':pc2_new, 'flow':flow_new, 'mask1_tracks_flow':mask_pc1_new, 'mask2_tracks_flow':mask_pc2_new}
        #np.save(osp.join(target_dir, 't9', split, all_filenames[i]), data)

        # # t10: crop
        pc1_new, pc2_new, mask_pc1_new, mask_pc2_new, flow_new = crop_data(pc1, pc2, flow, 0.5, 0.5, 0.5)
        data = {'pc1': pc1_new, 'pc2': pc2_new, 'flow': flow_new, 'mask1_tracks_flow': mask_pc1_new,
                'mask2_tracks_flow': mask_pc2_new}
        np.save(osp.join(target_dir, 't10', split, all_filenames[i]), data)

        # # t11: fov
        # fov_data(pc1, pc2, flow, 40)
        # data = {'pc1': pc1_new, 'pc2': pc2_new, 'flow': flow_new, 'mask1_tracks_flow': mask_pc1_new,
        #         'mask2_tracks_flow': mask_pc2_new}
        # np.save(osp.join(target_dir, 't11', split, all_filenames[i]), data)

        # visualize
        file_name = 't11'
        visualize_pc_and_flow(pc1_new, pc2_new, flow_new, save_dir, file_name, pc_range=new_range)
        break


if __name__=='__main__':
    main()