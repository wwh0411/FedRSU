import torch
from torch import optim
from tqdm import tqdm
import sys
import copy
# sys.path.append('../utils')
# from utils import to_device
# sys.path.append('../losses')
# from losses import build_criterion
# from fed_algs.alg_util import *

import numpy as np
import sympy as sp

def jacobian(f, x):
    a, b = np.shape(x)  # 变量个数
    x1, x2 = sp.symbols('x1 x2')  # 定义变量
    x3 = [x1, x2]
    a1 = np.zeros((b, b))  # 矩阵存放hesse矩阵的值
    for i in range(b):
        for j in range(b):
            a1[i, j] = sp.diff(f, x3[i], x3[j]).subs({x1: x[0][0], x2: x[0][1]})  # 函数求导，变量赋值

    return a1


def local_train_fednew(configs, args, train_dls, round, clients_this_round, local_model_list, global_model, writer, device,
                   local_auxiliary_list=None, global_auxiliary=None):

    # Sync model parameters
    global_w = global_model.state_dict()
    for model in local_model_list:
        model.load_state_dict(global_w)

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
        model.train()
        itrs = 0
        for epoch in range(configs['fed_params']['num_local_epochs']):
            for i, batch_data in tqdm(enumerate(train_loader)):
                itrs += 1
                batch_data = to_device(batch_data, device)
                # print(batch_data)
                batch_data['output'] = model(batch_data, configs)

                loss = criterion(batch_data, configs)
                # loss = criterion(pc1, pc2, flows_pred)
                loss = loss / configs['accumulation_step']
                loss.backward()

                for name, para in model.named_parameters():
                    grad = para.grad(retain_graph=True, create_graph=True)

                        # torch.autograd.grad(f, x, retain_graph=True, create_graph=True)
                    # print(grad)
                    print(type(grad))
                # 定义Print数组,为输出和进一步利用Hessian矩阵作准备
                    hess = torch.tensor([])

                    for anygrad in grad[0]:  # torch.autograd.grad返回的是元组
                        hess = torch.cat((hess, torch.autograd.grad(anygrad, retain_graph=True)[0]))


                if (itrs + 1) % configs['accumulation_step'] == 0:
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()


                if args.debug:
                    break
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, global_step=round)
            scheduler.step()
        model.to('cpu')


def fednew():
    # x = torch.tensor([0., 0, 0], requires_grad=True)
    # b = torch.tensor([1., 3, 5])
    # A = torch.tensor([[-5, -3, -0.5], [-3, -2, 0], [-0.5, 0, -0.5]])
    # f = b @ x + 0.5 * x @ A @ x

    x = torch.tensor([0., 0], requires_grad=True)
    b = torch.tensor([-4., 0])
    A = torch.tensor([[2., 0], [0, 1]])
    f = b @ x + x @ A @ x
    # 计算一阶导数,因为我们需要继续计算二阶导数,所以创建并保留计算图
    grad = torch.autograd.grad(f, x, retain_graph=True, create_graph=True)
    print(grad)
    # 定义Print数组,为输出和进一步利用Hessian矩阵作准备
    Print = torch.tensor([])

    for anygrad in grad[0]:  # torch.autograd.grad返回的是元组
        Print = torch.cat((Print, torch.autograd.grad(anygrad, x, retain_graph=True)[0]))

    print(Print.view(x.size()[0], -1))

    x1, x2 = sp.symbols('x1 x2')
    print(type(x1))
    x = [[0, 1]]
    f = 2 * x1 ** 2 + x2 ** 2 - 4 * x1 + 2

    print(jacobian(f, x))



fednew()




# class FedLogitCal(FedAvg):
#
#
#     def _local_training(self, party_id):
#         model = self.party2nets[party_id]
#         model.train()
#         model.cuda()
#
#         train_dataloader = self.party2loaders[party_id]
#         test_dataloader = self.test_dl
#
#         # collect number of data per class on current local client
#         n_class = model.classifier.out_features
#         class2data = torch.zeros(n_class)
#         ds = train_dataloader.dataset
#         uniq_val, uniq_count = np.unique(ds.target, return_counts=True)
#         for i, c in enumerate(uniq_val.tolist()):
#             class2data[c] = uniq_count[i]
#         class2data = class2data.unsqueeze(dim=0).cuda()
#
#
#
#         train_acc, _ = compute_accuracy(model, train_dataloader)
#         test_acc, _ = compute_accuracy(model, test_dataloader)
#
#
#
#         optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
#                               lr=self.args.lr, momentum=self.args.rho, weight_decay=self.args.weight_decay)
#         criterion = nn.CrossEntropyLoss()
#
#         for epoch in range(self.args.epochs):
#             epoch_loss_collector = []
#             for batch_idx, (x, target) in enumerate(train_dataloader):
#                 x, target = x.cuda(), target.cuda()
#                 target = target.long()
#                 optimizer.zero_grad()
#                 out = model(x)
#
#                 # calibrate logit
#                 out -= self.appr_args.calibration_temp * class2data**(-0.25)
#
#                 loss = criterion(out, target)
#                 loss.backward()
#                 optimizer.step()
#
#                 epoch_loss_collector.append(loss.item())
#
#             epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
#             self.logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
#
#         model.to('cpu')
