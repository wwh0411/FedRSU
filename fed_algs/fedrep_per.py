import torch
from torch import optim
from tqdm import tqdm
import sys
import copy
sys.path.append('../utils')
from utils import to_device
sys.path.append('../losses')
from losses import build_criterion


def local_train_fedper(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer, device, local_auxiliary_list=None, global_auxiliary=None):

    global_para = global_model.state_dict()

    net_keys = [*global_para.keys()]
    num_layers = len(net_keys)

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    per_keys = []
    for key in net_keys:
        if key.split('.')[0] == 'flow_regressor':
            per_keys.append(key)


    # # 计算参数数目
    # if args.alg == 'fedrep' or args.alg == 'fedper':
    #     num_param_glob = 0
    #     num_param_local = 0
    #     for key in net_keys:
    #         num_param_local += net_keys.numel()
    #         print(num_param_local)
    #         if key in per_keys:
    #             num_param_glob += net_keys.numel()
    #     percentage_param = 100 * float(num_param_glob) / num_param_local
    #     print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
    #         num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))


    # training

    # Conduct local model training
    for client_id in clients_this_round:
        model = local_model_list[client_id].cuda()
        train_loader = train_dls[client_id]
        # === optimizer & scheduler setup ===
        optimizer = optim.Adam(model.parameters(), lr=configs['optimizer']['lr'],
                               weight_decay=configs['optimizer']['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=configs['scheduler']['milestones'],
                                                   gamma=configs['scheduler']['gamma'])
        loss_config = configs['loss']
        criterion = build_criterion(loss_config)

        # local rep initialization
        local_para = model.state_dict()
        for key in local_para:
            if key not in per_keys:
                local_para[key] = global_para[key]
        model.load_state_dict(local_para)

        model.train()
        itrs = 0
        for epoch in range(configs['fed_params']['num_local_epochs']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                itrs += 1
                batch_data = to_device(batch_data, device)
                # print(batch_data)
                batch_data['output'] = model(batch_data, configs)
                # print(batch_data['output'])
                loss = criterion(batch_data, configs)
                # loss = criterion(pc1, pc2, flows_pred)
                loss = loss / configs['accumulation_step']
                writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                if i % 50 == 0:
                    print(f'>> Round {round} | Client {client_id} | Iter {itrs} | Loss {loss}')
                    # break
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=round)
            scheduler.step()
        model.to('cpu')


def local_train_fedrep(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer, device, local_auxiliary_list=None, global_auxiliary=None):

    global_para = global_model.state_dict()

    net_keys = [*global_para.keys()]
    num_layers = len(net_keys)

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    per_keys = []
    for key in net_keys:
        if key.split('.')[0] == 'flow_regressor':
            per_keys.append(key)

    # print(num_layers)
    # print(net_keys)

    # training

    # Conduct local model training
    for client_id in clients_this_round:
        model = local_model_list[client_id].cuda()
        train_loader = train_dls[client_id]
        # === optimizer & scheduler setup ===
        optimizer = optim.Adam(model.parameters(), lr=configs['optimizer']['lr'],
                               weight_decay=configs['optimizer']['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=configs['scheduler']['milestones'],
                                                   gamma=configs['scheduler']['gamma'])
        loss_config = configs['loss']
        criterion = build_criterion(loss_config)

        # local rep initialization
        local_para = model.state_dict()
        for key in local_para:
            if key not in per_keys:
                local_para[key] = global_para[key]
        model.load_state_dict(local_para)

        for name, para in model.named_parameters():
            if name.split('.')[0] == 'flow_regressor':
                para.requires_grad = True
            else:
                para.requires_grad = False
        model.train()
        itrs = 0
        for epoch in range(configs['fed_params']['num_local_epochs']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                itrs += 1
                batch_data = to_device(batch_data, device)
                # print(batch_data)
                batch_data['output'] = model(batch_data, configs)
                # print(batch_data['output'])
                loss = criterion(batch_data, configs)
                # loss = criterion(pc1, pc2, flows_pred)
                loss = loss / configs['accumulation_step']
                writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                if i % 50 == 0:
                    print(f'>> Round {round} | Client {client_id} | Iter {itrs} | Loss {loss}')
                    # break

        # local update rep
        for name, para in model.named_parameters():
            if name.split('.')[0] == 'flow_regressor':
                para.requires_grad = False
            else:
                para.requires_grad = True
        for epoch in range(configs['fed_params']['num_local_epochs']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                itrs += 1
                batch_data = to_device(batch_data, device)
                # print(batch_data)
                batch_data['output'] = model(batch_data, configs)
                # print(batch_data['output'])
                loss = criterion(batch_data, configs)
                # loss = criterion(pc1, pc2, flows_pred)
                loss = loss / configs['accumulation_step']
                writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                if i % 50 == 0:
                    print(f'>> Round {round} | Client {client_id} | Iter {itrs} | Loss {loss}')
                    # break
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=round)
            scheduler.step()
        model.to('cpu')


def global_aggregate_fedper(configs, args, dataset_size_list, round, clients_this_round, local_model_list, global_model,
                            global_auxiliary=None, global_anxiliary2=None, local_auxiliary_list=None):

        global_para = global_model.state_dict()
        net_keys = [*global_para.keys()]
        num_layers = len(net_keys)

        # specify the representation parameters (in w_glob_keys) and head parameters (all others)
        per_keys = []
        for key in net_keys:
            if key.split('.')[0] == 'flow_regressor':
                per_keys.append(key)

        total_data_points = sum([dataset_size_list[c] for c in clients_this_round])
        global_w = global_model.state_dict()
        old_w = copy.deepcopy(global_w) if args.alg == 'fedavgm' else None
        for i, client_id in enumerate(clients_this_round):
            net_para = local_model_list[client_id].state_dict()
            for key in net_para:
                if key not in per_keys:
                    if i == 0:

                        global_w[key] = net_para[key] * dataset_size_list[client_id] / total_data_points
                    else:
                        global_w[key] += net_para[key] * dataset_size_list[client_id] / total_data_points
