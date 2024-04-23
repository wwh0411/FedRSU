import torch
from torch import optim
from tqdm import tqdm
import sys
import copy
sys.path.append('../utils')
from utils import to_device
sys.path.append('../losses')
from losses import build_criterion


def local_train_fedavg(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer, device, local_auxiliary_list=None, global_auxiliary=None):
    
    # Sync model parameters
    global_w = global_model.state_dict()
    for model in local_model_list:
        model.load_state_dict(global_w)

    # Conduct local model training
    for client_id in clients_this_round:
        model = local_model_list[client_id].cuda()
        if args.ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        train_loader = train_dls[client_id]
        # === optimizer & scheduler setup ===
        optimizer = optim.Adam(model.parameters(), lr=configs['optimizer']['lr'],
                                weight_decay=configs['optimizer']['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=configs['scheduler']['milestones'],
                                                    gamma=configs['scheduler']['gamma'])

        loss_config = configs['loss']
        if client_id < args.p:
            print('optical')
            loss_config['loss_type'] = 'unsup_optical'
        else:
            print('normal')
            loss_config['loss_type'] = 'unsup_l1_seq'
        criterion = build_criterion(loss_config)
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
                if args.local_rank == 0:
                    writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                if i % 50 == 0:
                    print(f'>> Round {round} | Client {client_id} | Iter {itrs} | Loss {loss}')
                if args.debug:
                    break
            current_lr = optimizer.param_groups[0]['lr']
            if args.local_rank == 0:
                writer.add_scalar('lr', current_lr, global_step=round)
            scheduler.step()
        if args.ddp:
            torch.distributed.barrier()
        model.to('cpu')


def local_train_fedprox(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer, device, local_auxiliary_list=None, global_auxiliary=None):
    
    # Sync model parameters
    global_w = global_model.state_dict()
    for model in local_model_list:
        model.load_state_dict(global_w)
    
    global_weight_collector = list(global_model.cuda().parameters())

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
        if client_id < args.p:
            print('optical')
            loss_config['loss_type'] = 'unsup_optical'
        else:
            print('normal')
            loss_config['loss_type'] = 'unsup_l1_seq'
        criterion = build_criterion(loss_config)
        model.train()
        itrs = 0
        for epoch in range(configs['fed_params']['num_local_epochs']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                itrs += 1
                batch_data = to_device(batch_data, device)
                
                batch_data['output'] = model(batch_data, configs)
                loss = criterion(batch_data, configs)

                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((configs['fed_params']['fedprox']['mu'] / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss = loss / configs['accumulation_step']
                writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                if i % 50 == 0:
                    print(f'>> Round {round} | Client {client_id} | Iter {itrs} | Loss {loss}')
                if args.debug:
                    break
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=round)
            scheduler.step()
        model.to('cpu')
    global_model.to('cpu')


def local_train_scaffold(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer, device, local_auxiliary_list=None, global_auxiliary=None):
    
    # Sync model parameters
    global_w = global_model.state_dict()
    for model in local_model_list:
        model.load_state_dict(global_w)

    # scaffold delta
    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    
    global_auxiliary.cuda()
    global_model.cuda()

    # Conduct local model training
    for client_id in clients_this_round:
        model = local_model_list[client_id].cuda()
        train_loader = train_dls[client_id]
        auxiliary_model = local_auxiliary_list[client_id].cuda()
        auxiliary_global_para = global_auxiliary.state_dict()
        auxiliary_model_para = auxiliary_model.state_dict()
        # === optimizer & scheduler setup ===
        optimizer = optim.Adam(model.parameters(), lr=configs['optimizer']['lr'],
                                weight_decay=configs['optimizer']['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=configs['scheduler']['milestones'],
                                                    gamma=configs['scheduler']['gamma'])
        loss_config = configs['loss']
        if client_id < args.p:
            print('optical')
            loss_config['loss_type'] = 'unsup_optical'
        else:
            print('normal')
            loss_config['loss_type'] = 'unsup_l1_seq'

        criterion = build_criterion(loss_config)
        model.train()
        itrs = 0
        scaf_lr = configs['fed_params']['scaffold']['lr']
        for epoch in range(configs['fed_params']['num_local_epochs']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                itrs += 1
                batch_data = to_device(batch_data, device)

                batch_data['output'] = model(batch_data, configs)
                loss = criterion(batch_data, configs)
                # loss = criterion(pc1, pc2, flows_pred)

                loss = loss / configs['accumulation_step']
                writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer.step()
                    net_para = model.state_dict()
                    for key in net_para:
                        net_para[key] = net_para[key] - optimizer.param_groups[0]['lr'] * (auxiliary_global_para[key] - auxiliary_model_para[key])
                    model.load_state_dict(net_para)
                    model.zero_grad()
                    optimizer.zero_grad()

                if i % 50 == 0:
                    print(f'>> Round {round} | Client {client_id} | Iter {itrs} | Loss {loss}')
                    # break
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=round)
            scheduler.step()

        # renew local c
        auxilary_new_para = auxiliary_model.state_dict()
        auxilary_delta_para = copy.deepcopy(auxiliary_model.state_dict())
        global_model_para = global_model.state_dict()
        # delta c
        net_para = model.state_dict()
        for key in net_para:
            auxilary_new_para[key] = auxilary_new_para[key] - auxiliary_global_para[key] + (global_model_para[key] - net_para[key]) / (itrs * scaf_lr)
            auxilary_delta_para[key] = auxilary_new_para[key] - auxiliary_model_para[key]
        auxiliary_model.load_state_dict(auxilary_new_para)
        auxiliary_model.to('cpu')

        for key in total_delta:
            total_delta[key] += auxilary_delta_para[key]
        model.to('cpu')
    global_model.to('cpu')

    for key in total_delta:
        total_delta[key] /= len(clients_this_round)
    # renew global c
    auxilary_global_para = global_auxiliary.state_dict()
    for key in auxilary_global_para:
        if auxilary_global_para[key].type() == 'torch.LongTensor':
            auxilary_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif auxilary_global_para[key].type() == 'torch.cuda.LongTensor':
            auxilary_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            auxilary_global_para[key] += total_delta[key]
    global_auxiliary.load_state_dict(auxilary_global_para)

def local_train_feddyn(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer, device, local_auxiliary_list=None, global_auxiliary=None):
    # Sync model parameters
    global_w = global_model.state_dict()
    for model in local_model_list:
        model.load_state_dict(global_w)

    global_model.cuda()

    # Conduct local model training
    for client_id in clients_this_round:
        model = local_model_list[client_id].cuda()
        train_loader = train_dls[client_id]
        auxiliary_model = local_auxiliary_list[client_id].cuda()
        previous_grads = auxiliary_model.state_dict()
        server_weights = global_model.state_dict()
        # TODO


def local_train_ditto(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer,
                       device, local_auxiliary_list=None, global_auxiliary=None):
    mu = configs['fed_params']['ditto']['mu']

    # sync local_global
    global_w = global_model.state_dict()
    for model in local_auxiliary_list:
        model.load_state_dict(global_w)

    local_w_list = []

    # Conduct local model training
    for client_id in clients_this_round:
        itrs = 0
        train_loader = train_dls[client_id]
        model_1 = local_auxiliary_list[client_id].cuda()

        # === optimizer & scheduler setup ===
        optimizer_1 = optim.Adam(model_1.parameters(), lr=configs['optimizer']['lr'],
                                 weight_decay=configs['optimizer']['weight_decay'])

        loss_config = configs['loss']
        if client_id < args.p:
            print('optical')
            loss_config['loss_type'] = 'unsup_optical'
        else:
            print('normal')
            loss_config['loss_type'] = 'unsup_l1_seq'
        criterion = build_criterion(loss_config)

        # activate model
        for params in model_1.parameters():
            params.requires_grad = True

        # update global w
        model_1.train()
        for epoch in range(configs['fed_params']['ditto']['ep_local']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                itrs += 1
                # model.zero_grad()
                # optimizer.zero_grad()
                batch_data = to_device(batch_data, device)

                batch_data['output'] = model_1(batch_data, configs)
                loss = criterion(batch_data, configs)
                # loss = criterion(pc1, pc2, flows_pred)
                loss = loss / configs['accumulation_step']
                writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:

                    optimizer_1.step()
                    model_1.zero_grad()
                    optimizer_1.zero_grad()
                # break

        # local update
        model = local_model_list[client_id].cuda()
        local_w = model.state_dict()
        local_w_list.append(local_w)
        optimizer_2 = optim.Adam(model.parameters(), lr=configs['optimizer']['lr'],
                                 weight_decay=configs['optimizer']['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_2,
                                                   milestones=configs['scheduler']['milestones'],
                                                   gamma=configs['scheduler']['gamma'])
        # Freeze model
        for params in model_1.parameters():
            params.requires_grad = False
        # activate model
        for params in model.parameters():
            params.requires_grad = True
        model.train()
        for epoch in range(configs['fed_params']['num_local_epochs']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                itrs += 1
                batch_data = to_device(batch_data, device)
                batch_data['output'] = model(batch_data, configs)
                loss = criterion(batch_data, configs)
                #
                global_para = list(model_1.parameters())
                fed_prox_reg = 0.0
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_para[param_index])) ** 2)
                loss += fed_prox_reg

                loss = loss / configs['accumulation_step']
                writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer_2.step()
                    model.zero_grad()
                    optimizer_2.zero_grad()
                # break
                # print loss
                if i % 50 == 0:
                    print(f'>> Client {client_id} | Epoch {round} | Iter {i} | Loss {loss}')
            current_lr = optimizer_2.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=round)
            scheduler.step()
        model.to('cpu')
        model_1.to('cpu')


def local_train_central(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer,
                       device, local_auxiliary_list=None, global_auxiliary=None):
    # Sync model parameters
    global_w = global_model.state_dict()

    # Conduct local model training
    if args.alg == 'central':
        clients_this_round = [0]
    elif args.alg == 'local':
        if args.num == -1:
            clients_this_round = clients_this_round
        else:
            clients_this_round = [args.num]
            print('train on :', args.num)
    # clients_this_round = [0]
    for client_id in clients_this_round:
        if args.alg == 'central':
            model = global_model.cuda()
        elif args.alg == 'local':
            model = local_model_list[client_id].cuda()
        # model = global_model.cuda()
        train_loader = train_dls[client_id]
        # === optimizer & scheduler setup ===
        optimizer = optim.Adam(model.parameters(), lr=configs['optimizer']['lr'],
                               weight_decay=configs['optimizer']['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=configs['scheduler']['milestones'],
                                                   gamma=configs['scheduler']['gamma'])
        loss_config = configs['loss']
        loss_config = configs['loss']
        if client_id < args.p:
            print('optical')
            loss_config['loss_type'] = 'unsup_optical'
        else:
            print('normal')
            loss_config['loss_type'] = 'unsup_l1_seq'
        criterion = build_criterion(loss_config)
        criterion = build_criterion(loss_config)
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
                if args.debug:
                    break

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=round)
            scheduler.step()
        model.to('cpu')


def local_train_perfed(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer,
                       device, local_auxiliary_list=None, global_auxiliary=None):
    pass
