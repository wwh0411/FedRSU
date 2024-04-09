import copy
import torch
from fed_algs.alg_util import *


def global_aggregate_fedavg(configs, args, dataset_size_list, round, clients_this_round, local_model_list, global_model,
                            global_auxiliary=None, global_anxiliary2=None, local_auxiliary_list=None):
    total_data_points = sum([dataset_size_list[c] for c in clients_this_round])
    # print(total_data_points)
    global_w = global_model.state_dict()
    # print(list(global_model.parameters())[100][0])
    # print(list(global_model.parameters())[20][0].tolist())
    old_w = copy.deepcopy(global_w) if args.alg == 'fedavgm' else None
    for i, client_id in enumerate(clients_this_round):
        net_para = local_model_list[client_id].state_dict()
        if i == 0:
            for key in net_para:
                global_w[key] = net_para[key] * dataset_size_list[client_id] / total_data_points
                # print(global_w[key])
                # exit()
        else:
            for key in net_para:
                global_w[key] += net_para[key] * dataset_size_list[client_id] / total_data_points

    if args.alg == 'fedavgm':
        delta_w = copy.deepcopy(global_w)
        global_auxiliary_w = global_auxiliary.state_dict()
        beta = configs['fed_params']['fedavgm']['beta']
        for key in delta_w:
            delta_w[key] = old_w[key] - global_w[key]
            global_auxiliary_w[key] = beta * global_auxiliary_w[key] + delta_w[key]
            global_w[key] = (old_w[key] - global_auxiliary_w[key]) if round == 0 else (old_w[key] - global_auxiliary_w[key] / (1 + beta))

    global_model.load_state_dict(global_w)
    # print(list(global_model.parameters())[100][0])
    # print(list(global_model.parameters())[20][0].tolist())



def global_aggregate_fednova(configs, args, dataset_size_list, round, clients_this_round, local_model_list, global_model,
                             global_auxiliary=None, global_anxiliary2=None, local_auxiliary_list=None):
    total_data_points = sum([dataset_size_list[c] for c in clients_this_round])
    global_w = global_model.state_dict()
    old_w = copy.deepcopy(global_w)

    tau_eff = 0
    for client_id in clients_this_round:
        tau_eff += dataset_size_list[client_id]**2 / configs['batch_size'] / configs['accumulation_step'] / total_data_points
    print(type(tau_eff))
    for i, client_id in enumerate(clients_this_round):
        tau_i = dataset_size_list[client_id] / configs['batch_size'] / configs['accumulation_step']
        net_para = local_model_list[client_id].state_dict()
        for key in net_para:
            a = global_w[key].type()

            update_tmp = (net_para[key] - old_w[key]) / tau_i
            if global_w[key].type() == 'torch.LongTensor':
                global_w[key] += (tau_eff * update_tmp * dataset_size_list[client_id] / total_data_points).type(torch.LongTensor)
            else:
                global_w[key] += tau_eff * update_tmp * dataset_size_list[client_id] / total_data_points
            b = global_w[key].type()
            print(b)
            if a != b:
                print('!!!!!')
                exit()
    global_model.load_state_dict(global_w)


def global_aggregate_central(configs, args, dataset_size_list, round, clients_this_round, local_model_list, global_model,
                             global_auxiliary=None, global_anxiliary2=None, local_auxiliary_list=None):
    pass


def global_aggregate_fedoptim(configs, args, dataset_size_list, round, clients_this_round, local_model_list, global_model,
                              global_auxiliary=None, global_auxiliary2=None, local_auxiliary_list=None):
    moment_first_rate = configs['fed_params']['fedoptim']['moment_first_rate']
    moment_second_rate = configs['fed_params']['fedoptim']['moment_second_rate']
    optim = configs['fed_params']['fedoptim']['optim']
    # initialize server momentum

    moment_first = global_auxiliary.state_dict()
    moment_second = global_auxiliary2.state_dict()

    tau = configs['fed_params']['fedoptim']['tau']
    lr = configs['fed_params']['fedoptim']['lr']

    # conduct server update
    total_data_points = sum([dataset_size_list[c] for c in clients_this_round])
    global_w = global_model.state_dict()
    old_w = copy.deepcopy(global_w)

    # global aggregate
    for i, client_id in enumerate(clients_this_round):
        net_para = local_model_list[client_id].state_dict()
        if i == 0:
            for key in net_para:
                global_w[key] = net_para[key] * dataset_size_list[client_id] / total_data_points
        else:
            for key in net_para:
                global_w[key] += net_para[key] * dataset_size_list[client_id] / total_data_points
    # global update
    delta_w = copy.deepcopy(global_w)
    for key in delta_w:
        delta_w[key] = global_w[key] - old_w[key]

        # update first order moment
        moment_first[key] = moment_first_rate * moment_first[key] + \
            (1 - moment_first_rate) * delta_w[key]

        # update second order moment
        if optim == 'adagrad':
            moment_second[key] = moment_second[key] + delta_w[key] ** 2
        elif optim == 'adam':
            moment_second[key] = moment_second_rate * moment_second[key] + \
                (1 - moment_second_rate) * (delta_w[key] ** 2)
        elif optim == 'yogi':
            moment_second[key] = moment_second[key] - \
                (1 - moment_second_rate) * (delta_w[key] ** 2) * \
                torch.sign(moment_second[key] - delta_w[key] ** 2)

        # update global model parameters
        if optim == 'gd':
            # server learning rate is fixed to 1.0 regardless
            global_w[key] = old_w[key] + moment_first[key]
        else:
            # else it's second order algorithms
            global_w[key] = old_w[key] + \
                lr * moment_first[key] / (tau + torch.sqrt(moment_second[key]))

    global_model.load_state_dict(global_w)
    global_auxiliary.load_state_dict(moment_first)
    global_auxiliary2.load_state_dict(moment_second)



def global_aggregate_pfedgraph(configs, args, dataset_size_list, round, clients_this_round, local_model_list, global_model,
                               global_auxiliary=None, global_anxiliary2=None, local_auxiliary_list=None):
    total_data_points = sum([dataset_size_list[c] for c in clients_this_round])
    # print(total_data_points)
    global_w = global_model.state_dict()
    previous_global_model = copy.deepcopy(global_w)
    graph_matrix = global_auxiliary
    dw = global_anxiliary2
    for i, client_id in enumerate(clients_this_round):
        net_para = local_model_list[client_id].state_dict()
        if i == 0:
            for key in net_para:
                global_w[key] = net_para[key] * dataset_size_list[client_id] / total_data_points
                # print(global_w[key])
                # exit()
        else:
            for key in net_para:
                global_w[key] += net_para[key] * dataset_size_list[client_id] / total_data_points
    fed_avg_freqs =  {k: dataset_size_list[client_id] / total_data_points for k in clients_this_round}
    graph_matrix = update_graph_matrix_neighbor(graph_matrix, local_model_list, clients_this_round, global_w, dw, fed_avg_freqs,
                                                0.1,
                                                'all')  # Graph Matrix is not normalized yet
    aggregation_by_graph(configs, graph_matrix, local_model_list, clients_this_round, global_w,
                         local_auxiliary_list)  # Aggregation weight is normalized here

    global_model.load_state_dict(global_w)

def global_aggregate_fedexp(configs, args, dataset_size_list, round, clients_this_round, local_model_list, global_model,
                               global_auxiliary=None, global_anxiliary2=None, local_auxiliary_list=None):
    total_data_points = sum([dataset_size_list[c] for c in clients_this_round])
    global_w = global_model.state_dict()
    global_w_init = copy.deepcopy(global_w)
    global_w_update = copy.deepcopy(global_w)
    eps = 0.1
    for i, client_id in enumerate(clients_this_round):
        net_para = local_model_list[client_id].state_dict()
        if i == 0:
            for key in net_para:
                global_w_update[key] = net_para[key] * dataset_size_list[client_id] / total_data_points
        else:
            for key in net_para:
                global_w_update[key] += net_para[key] * dataset_size_list[client_id] / total_data_points



    # calculate eta_g, the server learning rate defined in FedExp
    # ======================================
    # calculate \|\bar{Delta}^{(t)}\|^{2}
    sqr_avg_delta = 0.0
    for key in global_w_update:
        sqr_avg_delta += ((global_w_update[key] - global_w_init[key]) ** 2).sum()

    # calculate \sum_{i}{p_{i}\|\Delta_{i}^{(t)}\|^{2}} for each client
    avg_sqr_delta = 0.0
    for i, client_id in enumerate(clients_this_round):
        net_para = local_model_list[client_id].state_dict()
        for key in net_para:
            avg_sqr_delta += dataset_size_list[i]/ total_data_points * ((net_para[key] - global_w_init[key]) ** 2).sum()

    eta_g = avg_sqr_delta / (2 * (sqr_avg_delta + eps))
    print(avg_sqr_delta, sqr_avg_delta)
    print(eta_g.item())
    eta_g = max(1.0, eta_g.item())
    print(eta_g)
    for key in global_w:
        global_w[key] = global_w_init[key] + eta_g * (global_w_update[key] - global_w_init[key])
    global_model.load_state_dict(global_w)