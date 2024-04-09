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
    parser.add_argument('--config', default='./config/zhan.yaml', type=str, help='Config files')
    parser.add_argument('--gpu', type=str, default='3', help='gpu')
    parser.add_argument('--alg', type=str, default='fedavg')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--downsize', action='store_true', default=False)
    parser.add_argument('--showhist', action='store_true', default=False)
    # device
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load config ===
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # === Init log ===
    save_dir, writer, csv_path = init_log(configs, args)

    # === Fix the random seed ===
    fix_seed(configs['random_seed'])

    # === Federated dataset setup ===
    train_dls, val_dls, dataset_size_list = get_dataloaders(configs, args)

    if args.downsize:
        configs['scene'] == 'split'
    #     for i in range(7, 17):
    #         train_loader = train_dls[i]
    #         l = len(train_loader)
    #         print(l)
    #         l = int(l/2)
    #         train_dls[i] = train_dls[i][:l]
    num_train_client, num_val_client = len(train_dls), len(val_dls)
    print('num_train_client: %d, num_val_client: %d' % (num_train_client, num_val_client))
    print([len(x) for x in train_dls])
    print([len(y) for y in val_dls])
    # for data in val_dls[7]:
    #     print(data)

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
    best_epe_ft = [[1 for _ in range(num_val_client)] for j in range(num_train_client)]
    best_loss_ft = [[100 for _ in range(num_val_client)] for j in range(num_train_client)]
    # === Training ===
    print('=====training=====:')
    # fed
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

        # print(list(global_model.parameters())[20][0].tolist())
        # evaluate
        if round % configs['eval_freq'] == 0:
            if configs['evaluation_mode'] == 'ft':
                epe_best, accr_best, accs_best, out_best = evaluate_local(configs, args, round, local_model_list, global_model, val_dls, \
                               device, writer, csv_path, save_dir, epe_best, accr_best, accs_best, out_best)
            else:
                epe_best, accr_best, accs_best, out_best, epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei, epe_best_7 \
                    = evaluate(configs, args, round, local_model_list, global_model, val_dls,
                     device, writer, csv_path, save_dir, epe_best, accr_best, accs_best, out_best,
                               epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei, epe_best_7)



    print('Complete %s training on %s for %d rounds %d epochs' % (args.alg, configs['data']['train_dataset_path'], configs['fed_params']['num_rounds'], configs['fed_params']['num_local_epochs']))

if __name__ == '__main__':
    main()