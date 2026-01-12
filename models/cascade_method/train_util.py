##########################################################
# Training basic codes
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################


import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import pandas as pd
import torch 
from tqdm import tqdm
import random
import torch
import multiprocessing

import warnings
warnings.filterwarnings("ignore")

from utils.metrics_tools import compute_metrics
from utils.plot_func import plot_log, wiggle_plot, heatmap_plot
from models.cascade_method.get_curve import split_group, interp_curves
from models.cascade_method.gauss_reg import posterior_regression
from models.cascade_method.clu_curves import clustering_main

"""
A tool for early stopping of training 
"""

# setup seed 
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)  # cpu 
    torch.cuda.manual_seed(seed)  # gpu 
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy


class EarlyStopping():
    def __init__(self, patience=7, delta=0, metric_name='Seg-MIoU', increase_better=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.EarlyStop = False
        self.delta = delta
        self.metric_name = metric_name
        self.increase_better = increase_better

    def __call__(self, metric_value, model, TrainParaDict, path):
        whether_better = 0
        if self.increase_better:
            score = metric_value
        else:
            score = -metric_value
            
        if self.best_score is None:  # init score
            self.SaveCheckpoint(score, model, TrainParaDict, path)
            self.best_score = score
            whether_better = 1

        elif score < self.best_score + self.delta:  # current is bad
            self.counter+=1
            print(f'#S EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.EarlyStop = True

        else:  # current is better
            self.SaveCheckpoint(score, model, TrainParaDict, path)
            self.best_score = score
            self.counter = 0
            whether_better = 1

        return whether_better
            
    def SaveCheckpoint(self, score, model, TrainParaDict, path): 
        if self.best_score is None:
            self.best_score = 0
        if self.increase_better:
            print(
                '#S %s increase (%.6f --> %.6f).  Saving model ...' % (self.metric_name, self.best_score, score)
            )
        else:
            print(
                '#S %s decrease (%.6f --> %.6f).  Saving model ...' % (self.metric_name, -self.best_score, -score)
            )
        SaveDict = {
            'weights': model.state_dict(),
            'parameters': TrainParaDict
        }
        torch.save(SaveDict, path+'/'+'model_checkpoint.pth')


def single_task(info_dict_path, if_rm, metrics_root, pred_hyper_dict={'win_k': 3, 'bw_data': 5, 'bw_para': 50, 'valid_range': 50, 'min_len': 5, 'clu_eps': 4}):
    info_dict = np.load(info_dict_path, allow_pickle=True).item()
    seg_map = info_dict['seg']
    gth = info_dict['gth_msfeat'][0]
    curve_manual = info_dict['curve_manual']
    curve_auto = split_group(seg_map)
    curve_auto_interp = interp_curves(curve_auto)
    curve_auto_interp_cp = curve_auto_interp.copy()
    for name in curve_auto_interp_cp:
        if len(curve_auto_interp_cp[name]) < pred_hyper_dict['win_k']:
            curve_auto_interp.pop(name)
            
    # * smooth the curve
    infer_opt = posterior_regression(win_k=pred_hyper_dict['win_k'], bw_data=pred_hyper_dict['bw_data'], bw_para=pred_hyper_dict['bw_para'])
    field_auto, slope_dict = infer_opt.est_prior(curve_auto_interp, gth.shape, valid_range=pred_hyper_dict['valid_range'])
    curve_concat, labels = clustering_main(curve_auto_interp, slope_dict, eps=pred_hyper_dict['clu_eps'])
    curve_dict_smooth = infer_opt.infer_posterior(curve_concat, field_auto)
    curve_dict_smooth_cp = curve_dict_smooth.copy()
    for name in curve_dict_smooth_cp:
        if len(curve_dict_smooth[name]) < pred_hyper_dict['min_len']:
            curve_dict_smooth.pop(name)
                            
    info_dict['curve_auto'] = curve_dict_smooth
    info_dict['curve_auto_raw'] = curve_auto
    info_dict['curve_auto_interp'] = curve_auto_interp
    info_dict['field_auto'] = field_auto
    info_dict['curve_cluster'] = {'curve_dict': curve_concat, 'labels': labels}
    if len(curve_manual) > 0:
        field_manual, _ = infer_opt.est_prior(curve_manual, gth.shape, valid_range=pred_hyper_dict['valid_range'])
    else:
        field_manual = []
    info_dict['field_manual'] = field_manual
    
    # * compute metrics
    if len(curve_manual) > 0:
        metrics_RCP = compute_metrics.RCP_metrics(gth, curve_dict_smooth, field_auto, curve_manual, field_manual, k=5, thre_rate=0.4)
        os.makedirs(metrics_root, exist_ok=True)
        met_save_path = os.path.join(metrics_root, os.path.basename(info_dict_path).replace('.npy', '-metrics.npy'))
        np.save(met_save_path, metrics_RCP)
    
    # * save & plot 
    if if_rm:
        if os.path.exists(info_dict_path):
            os.remove(info_dict_path)
    else:
        np.save(info_dict_path, info_dict)
        # training_loop.visual_step(fig_save_root, info_dict)
        
        
class training_loop:
    def __init__(
            self, 
            root_path,
            optimizer, 
            opt_strategy, 
            threshold,
            model, 
            train_dataloader, 
            valid_dataloader,
            criterion,
            max_epoch,
            early_stop_patience,
            config,
            device,
            seed,
            training_name,
            met_class=None, 
            valid_met_name='Seg-MIoU',
            if_print=1
            ):
        self.root_path = root_path
        self.optimizer = optimizer
        self.opt_strategy = opt_strategy
        self.threshold = threshold
        self.model = model
        self.tra_dl = train_dataloader
        self.val_dl = valid_dataloader
        self.criterion = criterion
        self.max_epoch = max_epoch
        self.config = config
        self.device = device
        self.early_stop = EarlyStopping(early_stop_patience, metric_name='Seg-MIoU', increase_better=True)
        self.seed = seed
        self.training_name = training_name
        self.valid_met_name = valid_met_name
        if met_class is None:
            self.met_seg = compute_metrics(threshold, device)
        else:
            self.met_seg = met_class
        self.if_print = if_print
        # ---- create training folder ----
        self.prepare_folder()

    def prepare_folder(self):
        for name in ['train', 'test']:
            path = os.path.join(self.root_path, name)
            if not os.path.exists(path):
                os.makedirs(path)
        self.model_path = os.path.join(self.root_path, 'train')  
        self.test_path = os.path.join(self.root_path, 'test')    
        self.save_folder = os.path.join(self.test_path, 'label')
        self.save_folder_nolabel = os.path.join(self.test_path, 'nolabel')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        if not os.path.exists(self.save_folder_nolabel):
            os.makedirs(self.save_folder_nolabel)
    
    def training_process(self):
        ############################################################
        # training process
        ############################################################
        # ---- start training ----
        print('# ------ training named %s start ------ #' % self.training_name)
        # ---- initial ----
        best_valid_info = ''# 初始化为空字符串，用于存储最佳验证信息
        best_valid_met = dict()#  初始化为空字典，用于存储最佳验证指标
        step_n = 0 # 初始化为 0，用于记录训练步数
        train_num = len(self.tra_dl)# 记录训练数据集的长度，用于在进度条中设置总步数
        log_iter_dict = {'train_loss': []}
        log_ep_dict = {'train_loss_avg': [], 'valid_loss_avg': []}
        
        # ---- training iteration ----
        for epoch in range(self.max_epoch):
            train_loss = []
            self.model.train()#111 将模型设置为训练模式
            #self.model.train的作用是将模型设置为训练模式，以便模型在训练阶段使用 dropout 和 batch normalization 等技巧
            #第一个 self.model.train() 语句确保了在训练循环开始时，模型处于训练模式
            if self.if_print:
                bar = tqdm(total=train_num, colour='#2a9d8f')
            for idx, (gth_LMO, gt_LMO, _, _) in enumerate(self.tra_dl):
                step_n += 1
                gth_LMO = gth_LMO.cuda(self.device)#输入数据和标签移动到指定的设备
                gt_LMO = gt_LMO.cuda(self.device)
                self.optimizer.zero_grad()#清除之前参数的梯度，以便进行新的参数更新
                seg_map, _ = self.model(gth_LMO)#使用模型 self.model 进行前向传播，得到预测的分割图像
                loss = self.criterion(seg_map, gt_LMO.to(torch.float32))
                train_loss.append(loss.item())#将当前批次的损失值添加到 train_loss 列表中
                loss.backward()#反向传播
                self.optimizer.step()#更新模型参数，优化器根据梯度更新模型参数
                train_loss_avg = np.average(train_loss)
                if self.if_print:
                    bar.set_description_str('Training | Epoch %d loss %.2e'%(epoch+1, train_loss_avg))
                    bar.update(1)
                # if idx >= 100:  # useful for debug to short time
                #     break
            if self.if_print:
                bar.close()
            # log the train loss
            log_iter_dict['train_loss'] += train_loss
            log_ep_dict['train_loss_avg'].append(train_loss_avg)
            
            # ---- validation process ----
            #self.valid_step返回值
            valid_loss_mean, val_met_dict, whether_better = self.valid_step()
            #检查优化器参数组中是否存在 'initial_lr' 键。如果不存在，则将当前学习率赋值给 lr_init 变量
            if 'initial_lr' not in self.optimizer.param_groups[0].keys():
                lr_init = self.optimizer.param_groups[0]['lr']
            else:
                lr_init = self.optimizer.param_groups[0]['initial_lr']
            valid_info = self.print_log(
                epoch+1, step_n, train_loss_avg, valid_loss_mean,
                self.optimizer.param_groups[0]['lr'], 
                lr_init, 
                val_met_dict
            )
            
            # log validation results
            log_ep_dict['valid_loss_avg'].append(valid_loss_mean)
            for met, values in val_met_dict.items():
                log_ep_dict.setdefault('valid_%s'%met, [])
                log_ep_dict['valid_%s'%met].append(values)
            if epoch == 0 or whether_better:
                best_valid_info = valid_info
                best_valid_met = val_met_dict
            for met, values in best_valid_met.items():
                log_ep_dict.setdefault('valid_best_%s'%met, [])
                log_ep_dict['valid_best_%s'%met].append(values)
            
            # turn to training mode
            self.model.train()
            # update the learning rate
            if self.opt_strategy is not None:#检查是否有指定的优化策略
                self.opt_strategy.step()#调用优化策略对象的 step 方法来更新模型的学习率
            
            # save epoch training log
            log_path = os.path.join(self.model_path, 'log')
            os.makedirs(log_path, exist_ok=True)
            epoch_log_df = pd.DataFrame(log_ep_dict)
            epoch_log_df.to_csv(os.path.join(log_path, 'epoch_log.csv'))
            plot_log(log_ep_dict, log_path)
            
            # whether break
            if self.early_stop.EarlyStop:
                # print final best valid results
                break  # break training
            
        # ---- training finished -----
        print('\n\n# ------ ending training ------ #\n\n')
        print(best_valid_info)

    def print_log(self, epoch, idx, 
                  train_loss, valid_loss, 
                  cur_lr, init_lr, val_met_dict):
        head_print = '####### validation epoch=%d step=%d #######\n' % (epoch, idx)
        loss_info = '# train_loss\t %.7f\t valid_loss\t %.7f\n' % (train_loss, valid_loss)
        lr_info = '# current_lr\t %.6f\t init_lr\t %.6f\n' % (cur_lr, init_lr)
        met_info = ''
        for name, met_list in val_met_dict.items():
            met_info += '# %s\t %.4f \n' % \
            (name, met_list)
        end_print = '#' * len(head_print) 
        valid_info = head_print+loss_info+lr_info+met_info+end_print+'\n\n'
        print(valid_info)
        return valid_info

    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            valid_met_log = {}
            valid_loss = []
            valid_num = len(self.val_dl)
            if self.if_print:
                bar = tqdm(total=valid_num, colour='#e76f51')
            compute_met = compute_metrics(threshold=self.threshold, device=self.device)
            for data_cell in self.val_dl:
                gth_LMO, gt_LMO, _, _ = data_cell
                gth_LMO = gth_LMO.cuda(self.device)
                gt_LMO = gt_LMO.cuda(self.device)               
                # model infer
                seg_map, _ = self.model(gth_LMO)
                val_loss = self.criterion(seg_map, gt_LMO.to(torch.float32))
                valid_loss.append(val_loss.item())               
                metrics = compute_met.compute(seg_map, gt_LMO)
                # 将计算得到的指标存储到 valid_metrics 字典中
                for metric_name, metric_value in metrics.items():
                    valid_met_log.setdefault(metric_name, [])
                    valid_met_log[metric_name].append(metric_value)    
                if self.if_print:
                    bar.set_description_str('Valid: loss %.2e'%np.average(valid_loss))
                    bar.update(1)
            if self.if_print:
                bar.close()
            
        # summary the results of validation
        valid_loss_mean = np.average(valid_loss)
        valid_met_avg = {met: np.average(value_list) for met, value_list in valid_met_log.items()}
            
        # judge to early stop
        whether_better = self.early_stop(
            valid_met_avg[self.valid_met_name], 
            model=self.model, 
            TrainParaDict=vars(self.config), 
            path=self.model_path
            )
        
        return valid_loss_mean, valid_met_avg, whether_better

    @staticmethod
    def visual_step(save_root, info_dict):
        # plot results
        id = info_dict['id']
        # gth = np.squeeze(info_dict['gth'])
        gth_ms = np.squeeze(info_dict['gth_msfeat'])
        gth_agc = gth_ms[1]
        seg = np.squeeze(info_dict['seg'])
        curve_manual = info_dict['curve_manual']
        curve_dict = info_dict['curve_auto']
        curve_dict_raw = info_dict['curve_auto_raw']
        field_auto = info_dict['field_auto']
        field_manual = info_dict['field_manual']
        
        # 1 seg ori
        heatmap_plot(seg, size=(3.5, 30), save_path=os.path.join(save_root, '%s-1-seg_ori.pdf' % id), cmap='binary')
        # 2 gth agc + auto points
        heatmap_plot(gth_agc, curve_dict=curve_dict, size=(3.5, 30), save_path=os.path.join(save_root, '%s-2-curve-gth_agc.pdf' % id), cmap='binary')
        # 3 4 wiggle agc + auto curve
        wiggle_plot(gth_agc, curve_dict=curve_dict, size=(3, 30), save_path=os.path.join(save_root, '%s-3-curve-wiggle_agc.pdf' % id))
        wiggle_plot(gth_agc, curve_dict=curve_dict_raw, size=(3, 30), save_path=os.path.join(save_root, '%s-4-curve-wiggle_agc_raw.pdf' % id))
        # 5 other feat
        for k in range(gth_ms.shape[0]):
            heatmap_plot(gth_ms[k], size=(3.5, 30), save_path=os.path.join(save_root, '%s-5-ms-feat-%d.pdf' % (id, k)), cmap='binary')
        # 6 wiggle agc + manual curve
        if len(curve_manual):
            wiggle_plot(gth_agc, curve_dict=curve_dict, curve_dict_m=curve_manual, size=(5, 30), save_path=os.path.join(save_root, '%s-6-manual-wiggle_agc.pdf' % id))
        # 7 slope field
        heatmap_plot(field_auto, size=(3.5, 30), save_path=os.path.join(save_root, '%s-7-slope_field_auto.pdf' % id), cmap='rainbow')
        if len(field_manual):
            heatmap_plot(field_manual, size=(3.5, 30), save_path=os.path.join(save_root, '%s-7-slope_field_manual.pdf' % id), cmap='rainbow')
            
        
    def test_step(self, test_dl, new_agc, test_name='test', save_sample=0, pred_hyper_dict={'win_k': 3, 'bw_data': 5, 'bw_para': 50, 'valid_range': 50, 'min_len': 5, 'clu_eps': 2, 'proc_num': 8}):
        print('\n\n# ------ start test ------ #')
        # save root 
        save_root = os.path.join(self.root_path, test_name)
        test_samp_save = os.path.join(self.root_path, test_name, 'samples')
        fig_root = os.path.join(self.root_path, test_name, 'fig')
        metrics_root = os.path.join(self.root_path, test_name, 'metrics')
        for path in [save_root, test_samp_save, fig_root, metrics_root]:
            os.makedirs(path, exist_ok=True)
        
        # 加载最佳模型
        model_path = os.path.join(self.model_path, 'model_checkpoint.pth')
        model_file = torch.load(model_path, map_location='cuda:%d'%self.device)
        self.model.load_state_dict(model_file['weights'])
        self.model.change_agc(new_agc, self.device)
        
        # 设置模型为评估模式
        self.model.eval()
        # 初始化测试指标
        save_path_list = []
        # 遍历测试数据集
        with torch.no_grad():
            test_metrics= {}
            test_num = len(test_dl)
            if self.if_print:
                bar = tqdm(total=test_num)
            compute_met = compute_metrics(threshold=self.threshold, device=self.device)
            for data_cell in test_dl:
                gth_LMO, gt_LMO, id_list, ori_shape = data_cell
                gth_LMO = gth_LMO.cuda(self.device)
                gt_LMO = gt_LMO.cuda(self.device)
                seg_map, gth_concat = self.model(gth_LMO)   
                # post-processing to velocity curve
                seg_map_np = seg_map.squeeze().detach().cpu().numpy()
                gth_np = gth_concat.squeeze().detach().cpu().numpy()
                
                if len(seg_map_np.shape) == 2:  # ensure bs = 1
                    seg_map_np = seg_map_np[np.newaxis, :, :]
                    gth_np = gth_np[np.newaxis, :, :]
                    
                for k, id in enumerate(id_list):
                    h, w = list(map(int, ori_shape[k].split('-')))
                    seg_map_np_k = seg_map_np[k, :h, :w] 
                    gth_np_ms = gth_np[k, :, :h, :w] 
                    try:
                        curve_manual = test_dl.dataset.get_manual_lab(id)
                        curve_manual = interp_curves(curve_manual)
                    except FileNotFoundError:
                        curve_manual = dict()
                    
                    # * save seg results
                    info_dict = {
                        'id': id, 
                        'gth_msfeat': gth_np_ms,
                        'seg': seg_map_np_k,
                        'curve_manual': curve_manual
                    }
                    save_path = os.path.join(test_samp_save, 'sample_%s.npy' % id)
                    np.save(save_path, info_dict)
                    save_path_list.append(save_path)
                        
                # compute metrics
                metrics_seg = compute_met.compute(seg_map, gt_LMO)
                for metric_name, metric_value in metrics_seg.items():
                    test_metrics.setdefault(metric_name, [])
                    test_metrics[metric_name].append(metric_value) 
                if self.if_print:   
                    bar.update(1)
                    bar.set_description_str('# Test Process')
            if self.if_print:
                bar.close()
        
        # * multiprocess to do the post-processing
        task_num = len(save_path_list)
        # hyper-parameters of pp
        pred_hyper_list = [pred_hyper_dict] * task_num
        if save_sample <= 0:
            save_list = [0] * task_num
        else:
            save_list = [0] * save_sample + [1] * (task_num - save_sample)
        metrics_list = [metrics_root] * task_num
        param_list = list(zip(save_path_list, save_list, metrics_list, pred_hyper_list))
        # 创建一个进程池
        pool = multiprocessing.Pool(processes=pred_hyper_dict['proc_num'])
        pool.starmap(single_task, param_list)
        pool.close()  # 关闭进程池
        pool.join()  # 等待所有进程结束
        # single_task(*param_list[0])
        # * concat the metrics of RCP
        for met_path in os.listdir(metrics_root):
            if os.path.exists(os.path.join(metrics_root, met_path)):
                metrics_RCP = np.load(os.path.join(metrics_root, met_path), allow_pickle=True).item()
                for metric_name, metric_value in metrics_RCP.items():
                    test_metrics.setdefault(metric_name, [])
                    test_metrics[metric_name].append(metric_value) 
                os.remove(os.path.join(metrics_root, met_path))
        if os.path.exists(metrics_root):
            os.rmdir(metrics_root)
            
        save_test_path = os.path.join(save_root, 'test_results.csv')
        test_mean_metrics = {met: [np.average(value_list)] for met, value_list in test_metrics.items()}
        test_df = pd.DataFrame(test_mean_metrics)
        test_df.to_csv(save_test_path, index=False)
        print('# ------ ending test ------ #\n\n')
        return test_df
        

