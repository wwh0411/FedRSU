from .flowstep3d import FlowStep3D
from .pointpwc import PointPWCNet
from .flownet import FlowNet3D
import copy
import torch

def build_model(configs, args, num_train_client):
    # set local model
    model_config = configs['model_params']
    if model_config['model_name'] == 'flowstep3d':
        global_model = FlowStep3D(npoint=model_config['npoint'],
                                  use_instance_norm=model_config['use_instance_norm'],
                                  loc_flow_nn=model_config['loc_flow_nn'],
                                  loc_flow_rad=model_config['loc_flow_rad'],
                                  k_decay_fact=model_config['k_decay_fact'])
    elif model_config['model_name'] == 'pointpwc':
        global_model = PointPWCNet()
    elif model_config['model_name'] == 'flownet':
        global_model = FlowNet3D()

    # model list
    local_model_list = [copy.deepcopy(global_model) for _ in range(num_train_client)]
    local_auxiliary_list = [copy.deepcopy(global_model) for _ in range(num_train_client)] if args.alg in ('scaffold', 'ditto', 'fedoptim', 'pfedme', 'pfedgraph') else None
    #local_auxiliary_list2 = [copy.deepcopy(global_model) for _ in range(num_train_client)] if args.alg in ('fedoptim') else None
    global_auxiliary = copy.deepcopy(global_model) if args.alg in ('scaffold', 'ditto', 'fedavgm', 'fedoptim','pfedme', 'pfedgraph') else None
    global_auxiliary2 = copy.deepcopy(global_model) if args.alg in ('fedoptim') else None

    if args.alg == 'fedavgm':
        global_auxiliary_w = global_auxiliary.state_dict()
        for key in global_auxiliary_w.keys():
            global_auxiliary_w[key].zero_()
        global_auxiliary.load_state_dict(global_auxiliary_w)
    if args.alg == 'fedoptim':
        global_aux_w = global_auxiliary.state_dict()
        global_aux_w2 = global_auxiliary2.state_dict()
        for key in global_aux_w:
            global_aux_w[key].zero_()
            global_aux_w2[key].zero_()
        global_auxiliary.load_state_dict(global_aux_w)
        global_auxiliary2.load_state_dict(global_aux_w2)
    if args.alg == 'pfedgraph':
        graph_matrix = torch.ones(num_train_client, num_train_client) / (num_train_client - 1)  # Collaboration Graph
        graph_matrix[range(num_train_client), range(num_train_client)] = 0
        global_auxiliary = graph_matrix
        dw = []
        for i in range(num_train_client):
            dw.append({key: torch.zeros_like(value) for key, value in local_model_list[i].named_parameters()})
        global_auxiliary2 = dw
    return global_model, local_model_list, local_auxiliary_list, global_auxiliary, global_auxiliary2