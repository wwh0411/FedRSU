import os
import os.path as osp
from tensorboardX import SummaryWriter
import socket
import yaml
import csv
import time
import random
import numpy as np
import torch


def init_log(configs, args):
    '''
    Set up the log directory / tensorboard writer / csv file
    '''
    log_path = os.path.join('logs', configs['tag'])
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    save_dir = os.path.join(log_path, args.alg + '_'
                            + str(configs['fed_params']['sample_fraction'])
                            + configs['evaluation_mode'] +'_'+ str(args.num)
                            + '_' + time.strftime('%d%B%Yat%H_%M_%S%Z'))

    writer = SummaryWriter(save_dir)
    with open(osp.join(save_dir, 'config.yaml'), mode='w') as f:
        yaml.dump(configs, f)

    # csv file
    csv_path = save_dir + '/result.csv'
    with open(csv_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['round', 'client', 'epe3d', 'loss', 'best_epe', 'best_loss', 'best_accs', 'best_accr', 'best_out'])

    return save_dir, writer, csv_path


# def fix_seed(seed):
#     '''
#     Fix the random seed
#     '''
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False