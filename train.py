import os
import yaml
import argparse
import copy
import random

import torch
from datasets import get_dataloaders
from models import build_model
from fed_algs import alg2local, alg2global
from utils import init_log, fix_seed, evaluate, evaluate_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/lumpi_flowstep3d_gen_seen.yaml', type=str, help='Config files')
    parser.add_argument('--gpu', type=str, default='7', help='assign gpu index')
    parser.add_argument('--alg', type=str, default='central', help="training algorithms")
    parser.add_argument('--debug', action='store_true', default=False, help="test if code works, it will set iter_steps to 1")
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=0, help="for distributed learning")
    parser.add_argument("--p", type=int, default=7, help="num of clients that use optical loss")

    args = parser.parse_args()

    # device
    if args.ddp:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device('cuda', args.local_rank)
        world_size = torch.distributed.get_world_size()
        args.world_size = world_size
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load config ===
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # === Init log ===
    if args.local_rank == 0:
        save_dir, writer, csv_path = init_log(configs, args)
    else:
        save_dir, writer, csv_path = None, None, None

    # === Fix the random seed ===
    fix_seed(configs['random_seed'])

    # === Federated dataset setup ===
    train_dls, val_dls, dataset_size_list = get_dataloaders(configs, args)
    num_train_client, num_val_client = len(train_dls), len(val_dls)
    print('num_train_client: %d, num_val_client: %d' % (num_train_client, num_val_client))
    print([len(x) for x in train_dls])
    print([len(y) for y in val_dls])


    # === Federated model setup ===
    global_model, local_model_list, local_auxiliary_list, global_auxiliary, global_auxiliary2 = build_model(configs, args, num_train_client)

    # === Load from checkpoint ===
    if configs['load_model'] != "":
        pass

    # === Init records ===
    best_epe, best_loss = [1 for _ in range(num_val_client)], [100 for _ in range(num_val_client)]
    best_accs, best_accr = [0.0 for _ in range(num_val_client)], [0.0 for _ in range(num_val_client)]
    best_out = [1 for _ in range(num_val_client)]
    epe_best, accr_best, accs_best, out_best = 1.0, 0.0, 0.0, 1.0
    epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei = 1.0, 0.0, 0.0, 1.0
    epe_best_7 = 1.0

    # === Training ===
    print('=====training=====:')
    # fed main
    local_train = alg2local[args.alg]
    global_aggregate = alg2global[args.alg]
    for round in range(configs['fed_params']['num_rounds']):
        # sanmple
        clients_this_round = random.sample([i for i in range(num_train_client)], int(num_train_client*configs['fed_params']['sample_fraction']))
        print('this round:', clients_this_round)
        # local update
        local_train(configs, args, train_dls, round, clients_this_round, \
                    local_model_list, global_model, writer, device, \
                    local_auxiliary_list, global_auxiliary)
        # global aggregate
        global_aggregate(configs, args, dataset_size_list, round, clients_this_round, \
                            local_model_list, global_model, global_auxiliary, global_auxiliary2, local_auxiliary_list)
        # evaluate
        print('eval:')
        # evaluate
        if round % configs['eval_freq'] == 0:
            if configs['evaluation_mode'] == 'ft':
                epe_best, accr_best, accs_best, out_best = evaluate_local(configs, args, round, local_model_list,
                                                                          global_model, val_dls, \
                                                                          device, writer, csv_path, save_dir, epe_best,
                                                                          accr_best, accs_best, out_best)
            else:
                epe_best, accr_best, accs_best, out_best, epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei, epe_best_7 \
                    = evaluate(configs, args, round, local_model_list, global_model, val_dls,
                               device, writer, csv_path, save_dir, epe_best, accr_best, accs_best, out_best,
                               epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei, epe_best_7)

    print('Complete %s training on %s for %d rounds %d epochs' % (args.alg, configs['data']['train_dataset_path'], configs['fed_params']['num_rounds'], configs['fed_params']['num_local_epochs']))


if __name__ == '__main__':
    main()