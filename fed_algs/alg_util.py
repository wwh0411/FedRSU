import torch
import numpy as np
import copy
import cvxpy as cp

def weight_flatten(model):
    params = []
    for k in model:
        # if 'classifier' in k:
        if 'fc' in k:
        # if 'conv' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def cal_model_cosine_difference(local_model_list, clients_this_round, initial_global_parameters, dw, similarity_matric):
    model_similarity_matrix = torch.zeros((len(clients_this_round),len(clients_this_round)))

    for i, client_id in enumerate(clients_this_round):
        model_i = local_model_list[client_id].state_dict()

        for key in dw[client_id]:
            dw[client_id][key] = model_i[key] - initial_global_parameters[key]
    for i in range(len(local_model_list)):
        for j in range(i, len(local_model_list)):
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[clients_this_round[i]]).unsqueeze(0), weight_flatten_all(dw[clients_this_round[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[clients_this_round[i]]).unsqueeze(0), weight_flatten(dw[clients_this_round[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
    #             model_similarity_matrix[j, i] = diff
    # print("model_similarity_matrix" ,model_similarity_matrix)
    return model_similarity_matrix


def update_graph_matrix_neighbor(graph_matrix, local_model_list, clients_this_round, initial_global_parameters, dw, fed_avg_freqs, lambda_1, similarity_matric):
    # index_clientid = torch.tensor(list(map(int, list(nets_this_round.keys()))))     # for example, client 'index_clientid[0]'s model difference vector is model_difference_matrix[0]

    # model_difference_matrix = cal_model_difference(index_clientid, nets_this_round, nets_param_start, difference_measure)
    model_difference_matrix = cal_model_cosine_difference(local_model_list, clients_this_round, initial_global_parameters, dw, similarity_matric)
    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, clients_this_round, model_difference_matrix, lambda_1, fed_avg_freqs)
    # print(f'Model difference: {model_difference_matrix[0]}')
    # print(f'Graph matrix: {graph_matrix}')
    return graph_matrix


def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )
        prob.solve()

        graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
    return graph_matrix


def aggregation_by_graph(cfg, graph_matrix, local_model_list, clients_this_round, global_w, cluster_models):
    tmp_client_state_dict = {}
    for client_id in clients_this_round:
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

    for client_id in clients_this_round:
        tmp_client_state = tmp_client_state_dict[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        # if client_id==0:
        #     print(f'Aggregation weight: {aggregation_weight_vector}. Summation: {aggregation_weight_vector.sum()}')

        for neighbor_id in clients_this_round:
            net_para = local_model_list[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

    for client_id in clients_this_round:
        cluster_models[client_id].load_state_dict(tmp_client_state_dict[client_id])
        local_model_list[client_id].load_state_dict(tmp_client_state_dict[client_id])