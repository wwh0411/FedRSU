import torch
from torch import optim
from tqdm import tqdm
import sys
import copy
sys.path.append('../utils')
from utils import to_device
sys.path.append('../losses')
from losses import build_criterion
from fed_algs.alg_util import *


def local_train_pfedme(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer,
                       device, local_auxiliary_list=None, global_auxiliary=None):
    # Sync model parameters
    global_w = global_model.state_dict()
    for global_model_mimic in local_auxiliary_list:
        global_model_mimic.load_state_dict(global_w)
    lamb = configs['fed_params']['pfedme']['lambda']
    ita = configs['fed_params']['pfedme']['ita']
    # Conduct local model training
    for client_id in clients_this_round:
        model = local_model_list[client_id].cuda()
        train_loader = train_dls[client_id]
        global_model_mimic = local_auxiliary_list[client_id].cuda()

        # === optimizer & scheduler setup ===
        optimizer = optim.Adam(model.parameters(), lr=configs['optimizer']['lr'],
                               weight_decay=configs['optimizer']['weight_decay'])
        # optimizer = pFedMeOptimizer(model.parameters(), lr=configs['optimizer']['lr'], lamda=lamb)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=configs['scheduler']['milestones'],
                                                   gamma=configs['scheduler']['gamma'])
        loss_config = configs['loss']
        criterion = build_criterion(loss_config)
        # local param
        # local_params = copy.deepcopy(list(model.parameters()))
        # local_params = copy.deepcopy(model.parameters())
        model.train()
        itrs = 0
        for epoch in range(configs['fed_params']['num_local_epochs']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                global_mimic_collector = list(global_model_mimic.parameters())

                itrs += 1
                batch_data = to_device(batch_data, device)
                # print(batch_data)
                batch_data['output'] = model(batch_data, configs)
                # print(batch_data['output'])
                loss = criterion(batch_data, configs)
                # loss = criterion(pc1, pc2, flows_pred)

                # for fedprox
                fedme_reg = 0.0
                for param_index, param in enumerate(model.parameters()):
                    fedme_reg += ((lamb / 2) * torch.norm(
                        (param - global_mimic_collector[param_index])) ** 2)
                loss += fedme_reg


                loss = loss / configs['accumulation_step']
                writer.add_scalar('train_loss %d' % client_id, loss, global_step=itrs)
                loss.backward()

                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer.step()
                    # personalized_params = optimizer.step(local_params)
                    # for new_param, localweight in zip(personalized_params, local_params):
                    #     localweight.data = localweight.data - 1e-2 * lamb * (localweight.data - new_param.data)
                    global_mimic_w = global_model_mimic.state_dict()
                    local_w = model.state_dict()
                    for key in local_w:
                        global_mimic_w[key] = global_mimic_w[key] - ita * lamb * (global_mimic_w[key] - local_w[key])
                    global_model_mimic.load_state_dict(global_mimic_w)

                    model.zero_grad()
                    optimizer.zero_grad()
                scheduler.step()

                if i % 100 == 0:
                    print(f'>> Round {round} | Client {client_id} | Iter {itrs} | Loss {loss}')
                if args.debug:
                    break
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=round)
        # for param, new_param in zip(model.parameters(), local_params):
        #     param.data = new_param.data

        model.to('cpu')
        global_model_mimic.to('cpu')

def global_aggregate_pfedme(configs, args, dataset_size_list, round, clients_this_round, local_model_list, global_model,
                            global_auxiliary=None, global_auxiliary2=None, local_auxiliary_list=None):
    beta = configs['fed_params']['pfedme']['beta']
    total_data_points = sum([dataset_size_list[c] for c in clients_this_round])
    # print(total_data_points)
    global_w = global_model.state_dict()
    global_pre_w = copy.deepcopy(global_w)

    for i, client_id in enumerate(clients_this_round):
        local_aux_w = local_auxiliary_list[client_id].state_dict()
        if i == 0:
            for key in local_aux_w:
                global_w[key] = local_aux_w[key] * dataset_size_list[client_id] / total_data_points
        else:
            for key in local_aux_w:
                global_w[key] += local_aux_w[key] * dataset_size_list[client_id] / total_data_points

    for key in global_w:
        global_w[key] = (1 - beta) * global_pre_w[key] + beta * global_w[key]

    global_model.load_state_dict(global_w)
