import torch
import numpy as np
import os
from tqdm import tqdm
from collections import OrderedDict
import statistics
import sys
import csv
sys.path.append('../losses')
from losses import build_criterion
from .train_util import to_device
import matplotlib.pyplot as plt

class AverageMeter:
    def __init__(self):
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val, 1]})
            else:
                self.loss_dict[loss_name][0] += loss_val
                self.loss_dict[loss_name][1] += 1

    def get_mean_loss(self):
        all_loss_val = 0.0
        all_loss_count = 0
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            all_loss_val += loss_val
            all_loss_count += loss_count
        return all_loss_val / (all_loss_count / len(self.loss_dict))

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            loss_dict[loss_name] = loss_val / loss_count
        return loss_dict

    def get_printable(self):
        text = ""
        all_loss_sum = 0.0
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            all_loss_sum += loss_val / loss_count
            text += "(%s:%.4f) " % (loss_name, loss_val / loss_count)
        text += " sum = %.4f" % all_loss_sum
        return text


def visual_epe(epe_list, save_path):
    # epe_list = epe_norm.tolist()
    # print(np.array(epe_list).shape)
    plt.hist(epe_list, bins=20, rwidth=0.8, range=(0, 1), histtype='bar')

    # plt.show()
    # save_path = '/GPFS/data/wenhaowang/bev2/epe.png'
    plt.savefig(save_path, transparent=False, dpi=400)
    plt.close()
    # exit()


def eval_flow(batch_data, epe_norm_thresh=0.05, eps=1e-10):
    """
    Compute scene flow estimation metrics: EPE3D, Acc3DS, Acc3DR, Outliers3D.
    :param gt_flow: (B, N, 3) torch.Tensor.
    :param flow_pred: (B, N, 3) torch.Tensor.
    :param epe_norm_thresh: Threshold for abstract EPE3D values, used in computing Acc3DS / Acc3DR / Outliers3D and adapted to sizes of different datasets.
    :return:
        epe & acc_strict & acc_relax & outlier: Floats.
    """
    gt_flow = batch_data['flow_transformed'].detach().cpu()
    flow_pred = batch_data['output']['pred_flow'].detach().cpu()

    epe_norm = torch.norm(flow_pred - gt_flow, dim=2)
    sf_norm = torch.norm(gt_flow, dim=2)
    relative_err = epe_norm / (sf_norm + eps)
    epe = epe_norm.mean().item()
    # if True:
    #     visual_epe(epe_norm)
    # Adjust the threshold to the scale of dataset
    acc_strict = (torch.logical_or(epe_norm < epe_norm_thresh, relative_err < 0.05)).float().mean().item()
    acc_relax = (torch.logical_or(epe_norm < (2 * epe_norm_thresh), relative_err < 0.1)).float().mean().item()
    outlier = (torch.logical_or(epe_norm > (6 * epe_norm_thresh), relative_err > 0.1)).float().mean().item()
    return epe, acc_strict, acc_relax, outlier, epe_norm


def evaluate(configs, args, round, local_model_list, global_model, val_dls, device,
             writer, csv_path, save_dir, epe_best, accr_best, accs_best, out_best,
             epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei, epe_best_7):
    # global_model.cuda()
    loss_config = configs['loss']
    criterion = build_criterion(loss_config)
    mode = configs['evaluation_mode']

    model = global_model
    # print(list(global_model.parameters())[20][0].tolist())
    # print(list(model.parameters())[20][0].tolist())
    if mode != 'local':
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_round_%d.pth' % (round)))
    else:
        if mode == 'local':
            if args.num != -1:
                model = local_model_list[args.num]
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_round_%d.pth' % (round)))

    num_val_loaders = [len(x) for x in val_dls]
    epe_list = []
    accs_list = []
    accr_list = []
    out_list = []

    for idx, val_loader in enumerate(val_dls):
        valid_ave_loss = []
        eval_meter = AverageMeter()
        # set to local model
        if mode == 'local':
            if args.num != -1:
                model = local_model_list[args.num]
            else:
                model = local_model_list[idx]

        # evaluate run
        model.cuda()
        epe_norm_list = []
        with torch.no_grad():

            for i, batch_data in tqdm(enumerate(val_loader)):
                model.eval()

                batch_data = to_device(batch_data, device)

                batch_data['output'] = model(batch_data, configs)
                # print(batch_data['output'])
                # print(len(batch_data['output']))
                # a = batch_data['output']['flows']
                # # print(a)
                # for ten in a:
                #     ten.zero_()
                # # print(a)
                # batch_data['output']['flows'] = a
                # loss = criterion(batch_data, configs)
                # valid_ave_loss.append(loss.item())
                epe, acc_strict, acc_relax, outlier, epe_norm = eval_flow(batch_data)
                epe_norm_list.extend(epe_norm.tolist())
                eval_meter.append_loss({'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})

                if args.debug:
                    # if idx != 8:
                    break
        # valid_ave_loss = statistics.mean(valid_ave_loss)
        # writer.add_scalar('Validate_Loss %d' % idx, valid_ave_loss, round)
        # if args.showhist:
        #     visual_epe(epe_norm_list, os.path.join(save_dir, "{}_{}_{}".format(round, idx, 'epe.png')))
        # Accumulate evaluation results
        eval_avg = eval_meter.get_mean_loss_dict()
        if 'EPE' not in eval_avg.keys():
            eval_avg['EPE'] = float(1)
        if eval_avg['EPE'] != eval_avg['EPE']:
            eval_avg['EPE'] = float(1)
        writer.add_scalar('epe3d %d' % val_dls.index(val_loader), eval_avg['EPE'], round)
        epe3d = eval_avg['EPE']
        epe_list.append(epe3d)
        accs = eval_avg['AccS']
        accs_list.append(accs)
        accr = eval_avg['AccR']
        accr_list.append(accr)
        outlier = eval_avg['Outlier']
        out_list.append(outlier)

        # write best results
        # if epe3d < best_epe[idx]:
        #     best_epe[idx] = epe3d
        #     # writer.add_scalar('best_epe %d' % idx, best_epe[idx], 0)
        # if accs > best_accs[idx]:
        #     best_accs[idx] = accs
        # if accr > best_accr[idx]:
        #     best_accr[idx] = accr
        # if outlier < best_out[idx]:
        #     best_out[idx] = outlier
        with open(csv_path, 'a+') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([round, idx, valid_ave_loss, epe3d, accs, accr, outlier])

        print('| Round %d | val on %d | EPE %.5f | ' % (round, idx, epe3d))

        model.to('cpu')

    # get mean epe
    if mode in []:#'gen-seen','unseen']:
        epe_mean = sum([epe_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        epe_mean2 = sum(epe_list) / len(num_val_loaders)
        accr_mean = sum([accr_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        accs_mean = sum([accs_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        out_mean = sum([out_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
    else:
        # personalized
        epe_mean = sum(epe_list) / len(num_val_loaders)
        epe_mean_7 = sum(epe_list[:7]) / 7
        accr_mean = sum(accr_list) / len(num_val_loaders)
        accs_mean = sum(accs_list) / len(num_val_loaders)
        out_mean = sum(out_list) / len(num_val_loaders)
        epe_mean_wei = sum([epe_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        accr_mean_wei = sum([accr_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        accs_mean_wei = sum([accs_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        out_mean_wei = sum([out_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)

    # get best result

    if epe_mean < epe_best:
        epe_best = epe_mean
        accr_best = accr_mean
        accs_best = accs_mean
        out_best = out_mean
        epe_best_wei = epe_mean_wei
        accr_best_wei = accr_mean_wei
        accs_best_wei = accs_mean_wei
        out_best_wei = out_mean_wei

        epe_best_7 = epe_mean_7
        # best_epe = epe_list
        # best_accs = accs_list
        # best_accr = accr_list
        # best_out = out_list
    # else:
    #     # personal eval
    #     for i, epe in enumerate(epe_list):
    #         if epe < best_epe[i]:
    #             best_epe[i] = epe
    #             best_accs[i] = accs_list[i]
    #             best_accr[i] = accr_list[i]
    #             best_out[i] = out_list[i]
    #     epe_best = sum([epe_best[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)


    with open(csv_path, 'a+') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['mean result: ', round, epe_mean, accr_mean, accs_mean, out_mean,
                             'weighted:', epe_mean_wei, accr_mean_wei, accs_mean_wei, out_mean_wei,
                             'dair:', epe_mean_7])
        csv_writer.writerow(['best result: ', round, epe_best, accr_best, accs_best, out_best,
                             'weighted:', epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei,
                             'dair:', epe_best_7])

    writer.add_scalar('epe3d mean', epe_mean, round)
    print('mean result:', round, epe_mean, accr_mean, accs_mean, out_mean,
          'weighted:', epe_mean_wei, accr_mean_wei, accs_mean_wei, out_mean_wei)
    print('best result:', round, epe_best, accr_best, accs_best, out_best,
          'weighted:', epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei,
          'dair:', epe_best_7)

    return epe_best, accr_best, accs_best, out_best, epe_best_wei, accr_best_wei, accs_best_wei, out_best_wei, epe_best_7


def evaluate_local(configs, args, round, local_model_list, global_model, val_dls,
                   device, writer, csv_path, save_dir,
                   epe_best, accr_best, accs_best, out_best):

    loss_config = configs['loss']
    criterion = build_criterion(loss_config)
    mode = configs['evaluation_mode']


    num_val_loaders = [len(x) for x in val_dls]

    epe_mean_list = []
    accr_mean_list = []
    accs_mean_list = []
    out_mean_list = []
    epe_mean_wei_list = []

    for idx_model, model in enumerate(local_model_list):
        # save local_model
        torch.save(model.state_dict(), os.path.join(save_dir, 'local_model%d_round%d.pth' % (idx_model, round)))
        epe_list = []
        accs_list = []
        accr_list = []
        out_list = []

        model.cuda()
        for idx, val_loader in enumerate(val_dls):
            valid_ave_loss = []
            eval_meter = AverageMeter()

            with torch.no_grad():

                for i, batch_data in tqdm(enumerate(val_loader)):
                    model.eval()

                    batch_data = to_device(batch_data, device)
                    batch_data['output'] = model(batch_data, configs)
                    loss = criterion(batch_data, configs)
                    valid_ave_loss.append(loss.item())
                    epe, acc_strict, acc_relax, outlier, epe_norm = eval_flow(batch_data)
                    eval_meter.append_loss({'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})
                    if args.debug:
                        break
            valid_ave_loss = statistics.mean(valid_ave_loss)
            writer.add_scalar('Validate_Loss %d' % idx, valid_ave_loss, round)

            # Accumulate evaluation results
            eval_avg = eval_meter.get_mean_loss_dict()

            writer.add_scalar('epe3d %d' % val_dls.index(val_loader), eval_avg['EPE'], round)
            epe3d = eval_avg['EPE']
            epe_list.append(epe3d)
            accs = eval_avg['AccS']
            accs_list.append(accs)
            accr = eval_avg['AccR']
            accr_list.append(accr)
            outlier = eval_avg['Outlier']
            out_list.append(outlier)

            # write best results
            # if valid_ave_loss < best_loss_ft[idx_model][idx]:
            #     best_loss_ft[idx_model][idx] = valid_ave_loss
            #     writer.add_scalar('local best_loss %d %d' % (idx_model, idx), best_loss_ft[idx_model][idx], 0)
            # if eval_avg['EPE'] < best_epe_ft[idx_model][idx]:
            #     best_epe_ft[idx_model][idx] = eval_avg['EPE']
            #     writer.add_scalar('local best_epe %d %d' % (idx_model, idx), best_epe_ft[idx_model][idx], 0)
            with open(csv_path, 'a+') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([round, 'localmodel ' + str(idx_model) + ' val on ' + str(idx), valid_ave_loss, epe3d, accs, accr, outlier])



            print('Round %d | Client %d | val on % d Loss %.5f |' % (round, idx_model, idx, valid_ave_loss))
            # print('Evaluation result:', eval_avg)
        epe_mean = sum(epe_list) / len(num_val_loaders)
        epe_mean_7 = sum(epe_list[:7]) / 7
        accr_mean = sum(accr_list) / len(num_val_loaders)
        accs_mean = sum(accs_list) / len(num_val_loaders)
        out_mean = sum(out_list) / len(num_val_loaders)
        epe_mean_wei = sum([epe_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        accr_mean_wei = sum([accr_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        accs_mean_wei = sum([accs_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)
        out_mean_wei = sum([out_list[i] * num_val_loaders[i] for i in range(len(val_dls))]) / sum(num_val_loaders)

        epe_mean_list.append(epe_mean)
        accr_mean_list.append(accr_mean)
        accs_mean_list.append(accs_mean)
        out_mean_list.append(out_mean)
        epe_mean_wei_list.append(epe_mean_wei)
        print('Evaluation result:', round, idx_model, epe_mean, accr_mean, accs_mean, out_mean,
              'weight', epe_mean_wei, accr_mean_wei, accs_mean, out_mean_wei)
        with open(csv_path, 'a+') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(epe_list)
            csv_writer.writerow(['Evaluation result:', round, idx_model, epe_mean, accr_mean, accs_mean, out_mean,
              'weight', epe_mean_wei, accr_mean_wei, accs_mean, out_mean_wei])
        model.to('cpu')
    epe_mean_last = sum(epe_mean_list) / len(epe_mean_list)
    accr_mean_last = sum(accr_mean_list) / len(epe_mean_list)
    accs_mean_last = sum(accs_mean_list) / len(epe_mean_list)
    out_mean_last = sum(out_mean_list) / len(epe_mean_list)
    if epe_mean_last < epe_best:
        epe_best = epe_mean_last
        accr_best = accr_mean_last
        accs_best = accs_mean_last
        out_best = out_mean_last
    with open(csv_path, 'a+') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['mean result: ', round, epe_mean_last, accr_mean_last, accs_mean_last, out_mean_last])
        csv_writer.writerow(['best result: ', round, epe_best, accr_best, accs_best, out_best])

    writer.add_scalar('epe3d mean', epe_mean_last, round)
    print('Evaluation result:', round, epe_mean_last, accr_mean_last, accs_mean_last, out_mean_last)
    print('Evaluation result:', round, epe_best, accr_best, accs_best, out_best)
    return epe_best, accr_best, accs_best, out_best