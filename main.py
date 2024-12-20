# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :main_chapman.py
# @Time      :2024/4/2 8:29
# @Author    :Chen
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch
import data_utils as du
from torch.utils.data import DataLoader
import model_utils as mu
import torch.optim as optim
from tqdm import tqdm
import copy
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def single_contrastive(phase, dataloader, model, optimizer, para_dict):
    running_loss = 0.0
    running_cl_loss = 0.0
    running_lead_loss = 0.0
    n = 0
    for inputs, labels, pids in dataloader[phase]:
        inputs, labels = inputs.to(para_dict['device']), labels.to(para_dict['device'])
        if "train" in phase:
            _, outputs = model(inputs)
            cl_loss = du.obtain_contrastive_loss(outputs)
            lead_loss = du.obtain_lead_loss(inputs, outputs)
            loss = 0.5*cl_loss + 0.5*lead_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                _, outputs = model(inputs)
                cl_loss = du.obtain_contrastive_loss(outputs)
                lead_loss = du.obtain_lead_loss(inputs, outputs)
                loss = 0.5*cl_loss + 0.5*lead_loss
        running_cl_loss += cl_loss
        running_lead_loss += lead_loss
        running_loss += loss
        n += 1
    epoch_loss = running_loss / n
    epoch_cl_loss = running_cl_loss / n
    epoch_lead_loss = running_lead_loss / n
    return epoch_loss.cpu().detach().numpy(), epoch_cl_loss.cpu().detach().numpy(), epoch_lead_loss.cpu().detach().numpy()


def single_linear(phase, dataloader, model, optimizer, para_dict):
    running_loss = 0.0
    n = 0
    outputs_list = []
    labels_list = []
    if para_dict['dataset_name'] == 'chapman':
        lf = torch.nn.CrossEntropyLoss()
    elif para_dict['dataset_name'] in ['physionet2020', 'ptb_xl']:
        lf = torch.nn.BCEWithLogitsLoss()
    for inputs, labels, pids in dataloader[phase]:
        inputs, labels = inputs.to(para_dict['device']), labels.to(para_dict['device'])
        if "train" in phase:
            outputs = model(inputs)
            loss = lf(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = lf(outputs, labels)

        outputs_list.append(outputs.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())
        running_loss += loss
        n += 1

    epoch_loss = running_loss / n
    outputs_list = np.concatenate(outputs_list)
    labels_list = np.concatenate(labels_list)
    ohe = LabelBinarizer()
    labels_ohe = ohe.fit_transform(labels_list)
    all_auc = []
    for i in range(labels_ohe.shape[1]):
        auc = roc_auc_score(labels_ohe[:, i], outputs_list[:, i])
        all_auc.append(auc)
    epoch_auroc = np.mean(all_auc)
    return epoch_loss.cpu().detach().numpy(), epoch_auroc


def cl_train(para_dict):
    # 训练无监督模型
    print('task: CL')
    phases = ['train', 'val']
    dataset = {phase: du.my_dataset_contrastive(phase, para_dict['basepath_to_data'], para_dict['dataset_name']) for phase in phases}
    dataloader = {phase: DataLoader(dataset[phase], batch_size=para_dict['batch_size'], shuffle=para_dict['shuffles'][phase], drop_last=False) for phase in phases}
    if para_dict['model_name'] == 'CNN':
        model = mu.CNN(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'AlexNet':
        model = mu.AlexNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'VGG':
        model = mu.VGG(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'ResNet':
        model = mu.ResNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    optimizer = optim.Adam(list(model.parameters()), lr=para_dict['lr'], weight_decay=0)
    stop_counter = 0
    best_loss = float('inf')
    with tqdm(range(para_dict['max_epochs']), dynamic_ncols=True) as tqdmEpochs:
        for epoch in tqdmEpochs:
            if stop_counter > para_dict['patience']:
                break
            for phase in phases:
                if phase == 'train':
                    model.train()
                elif phase == 'val':
                    model.eval()

                epoch_loss, epoch_cl_loss, epoch_lead_loss = single_contrastive(phase, dataloader, model, optimizer, para_dict)
                if phase == 'train':
                    train_loss = epoch_loss
                    train_cl_loss = epoch_cl_loss
                    train_lead_loss = epoch_lead_loss
                elif phase == 'val':
                    val_loss = epoch_loss
                    val_cl_loss = epoch_cl_loss
                    val_lead_loss = epoch_lead_loss

                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy((model.state_dict()))
                    torch.save(best_model_wts, os.path.join(para_dict['save_path_dir'], 'pretrained_weight'))
                    stop_counter = 0
                elif phase == 'val' and epoch_loss >= best_loss:
                    stop_counter += 1
            tqdmEpochs.set_postfix(ordered_dict={
                "l": '%.2f' % train_loss + ',%.2f' % train_cl_loss + ',%.2f' % val_loss + ',%.2f' % val_cl_loss,
                'b': '%.2f' % best_loss,
                's': stop_counter
            })


def linear_train(para_dict):
    # 训练线性评估模型
    print('task: Linear')
    phases = ['train', 'val']
    dataset = {phase: du.my_dataset_contrastive(phase, para_dict['basepath_to_data'],  para_dict['dataset_name']) for phase in phases}
    dataloader = {phase: DataLoader(dataset[phase], batch_size=para_dict['batch_size'], shuffle=para_dict['shuffles'][phase], drop_last=False) for phase in phases}

    if para_dict['model_name'] == 'CNN':
        model = mu.CNN(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'AlexNet':
        model = mu.AlexNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'VGG':
        model = mu.VGG(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'ResNet':
        model = mu.ResNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    model.load_state_dict(torch.load(os.path.join(para_dict['save_path_dir'], 'pretrained_weight')))
    for param in model.parameters():  # freeze representation weights
        param.requires_grad_(False)
    model.to(para_dict['device'])
    """ Load Second Model for Classification """
    model = mu.SeNet(model, para_dict['noutputs'], para_dict['embedding_dim']).to(para_dict['device'])
    optimizer = optim.Adam(list(model.parameters()), lr=para_dict['lr'], weight_decay=0)
    stop_counter = 0
    best_loss = float('inf')
    with tqdm(range(para_dict['max_epochs']), dynamic_ncols=True) as tqdmEpochs:
        for epoch in tqdmEpochs:
            if stop_counter > para_dict['patience']:
                break
            for phase in phases:
                if phase == 'train':
                    model.train()
                elif phase == 'val':
                    model.eval()
                epoch_loss, epoch_roc = single_linear(phase, dataloader, model, optimizer, para_dict)
                if phase == 'train':
                    train_loss = epoch_loss
                    train_roc = epoch_roc
                elif phase == 'val':
                    val_loss = epoch_loss
                    val_roc = epoch_roc

                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy((model.state_dict()))
                    torch.save(best_model_wts, os.path.join(para_dict['save_path_dir'], 'line_weight'))
                    stop_counter = 0
                elif phase == 'val' and epoch_loss >= best_loss:
                    stop_counter += 1
            tqdmEpochs.set_postfix(ordered_dict={
                "l": '%.2f' % train_loss + ',%.2f' % train_roc + ',%.2f' % val_loss + ',%.2f' % val_roc,
                'b': '%.2f' % best_loss,
                's': stop_counter
            })


def linear_test(para_dict):
    phases = ['test', ]
    dataset = {phase: du.my_dataset_contrastive(phase, para_dict['basepath_to_data'],  para_dict['dataset_name']) for phase in phases}
    dataloader = {phase: DataLoader(dataset[phase], batch_size=para_dict['batch_size'], shuffle=para_dict['shuffles'][phase], drop_last=False) for phase in phases}

    if para_dict['model_name'] == 'CNN':
        model = mu.CNN(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'AlexNet':
        model = mu.AlexNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'VGG':
        model = mu.VGG(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'ResNet':
        model = mu.ResNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    model = mu.SeNet(model, para_dict['noutputs'], para_dict['embedding_dim']).to(para_dict['device'])
    model.load_state_dict(torch.load(os.path.join(para_dict['save_path_dir'], 'line_weight')))
    optimizer = optim.Adam(list(model.parameters()), lr=para_dict['lr'], weight_decay=0)
    for phase in phases:
        model.eval()
        epoch_loss, epoch_roc = single_linear(phase, dataloader, model, optimizer, para_dict)
        print('%.4f' % epoch_loss, '%.4f' % epoch_roc)
    return epoch_roc


def finetune_train(para_dict):
    print('task: Finetune_' +para_dict['dataset_name'])
    phases = ['train', 'val']
    dataset = {phase: du.my_dataset_contrastive(phase, para_dict['basepath_to_data'],  para_dict['dataset_name']) for phase in phases}
    dataloader = {phase: DataLoader(dataset[phase], batch_size=para_dict['batch_size'], shuffle=para_dict['shuffles'][phase],drop_last=False) for phase in phases}
    if para_dict['model_name'] == 'CNN':
        model = mu.CNN(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'AlexNet':
        model = mu.AlexNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'VGG':
        model = mu.VGG(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'ResNet':
        model = mu.ResNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    model.load_state_dict(torch.load(os.path.join(para_dict['save_path_dir'], 'pretrained_weight')))
    model.to(para_dict['device'])
    model = mu.SeNet(model, para_dict['noutputs'], para_dict['embedding_dim']).to(para_dict['device'])
    optimizer = optim.Adam(list(model.parameters()), lr=para_dict['lr'], weight_decay=0)

    stop_counter = 0
    best_loss = float('inf')
    with tqdm(range(para_dict['max_epochs']), dynamic_ncols=True) as tqdmEpochs:
        for epoch in tqdmEpochs:
            if stop_counter > para_dict['patience']:
                break
            for phase in phases:
                if phase == 'train':
                    model.train()
                elif phase == 'val':
                    model.eval()
                epoch_loss, epoch_roc = single_linear(phase, dataloader, model, optimizer, para_dict)
                if phase == 'train':
                    train_loss = epoch_loss
                    train_roc = epoch_roc
                elif phase == 'val':
                    val_loss = epoch_loss
                    val_roc = epoch_roc

                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy((model.state_dict()))
                    torch.save(best_model_wts, os.path.join(para_dict['save_path_dir'], para_dict['dataset_name']+'_weight'))
                    stop_counter = 0
                elif phase == 'val' and epoch_loss >= best_loss:
                    stop_counter += 1
            tqdmEpochs.set_postfix(ordered_dict={
                "l": '%.2f' % train_loss + ',%.2f' % train_roc + ',%.2f' % val_loss + ',%.2f' % val_roc,
                'b': '%.2f' % best_loss,
                's': stop_counter
            })


def finetune_test(para_dict):
    # 训练线性评估模型
    phases = ['test', ]
    dataset = {phase: du.my_dataset_contrastive(phase, para_dict['basepath_to_data'], para_dict['dataset_name']) for phase in phases}
    dataloader = {phase: DataLoader(dataset[phase], batch_size=para_dict['batch_size'], shuffle=para_dict['shuffles'][phase], drop_last=False) for phase in phases}

    if para_dict['model_name'] == 'CNN':
        model = mu.CNN(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'AlexNet':
        model = mu.AlexNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'VGG':
        model = mu.VGG(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    elif para_dict['model_name'] == 'ResNet':
        model = mu.ResNet(embedding_dim=para_dict['embedding_dim'], device=para_dict['device']).to(
            para_dict['device'])
    model = mu.SeNet(model, para_dict['noutputs'], para_dict['embedding_dim']).to(para_dict['device'])
    model.load_state_dict(torch.load(os.path.join(para_dict['save_path_dir'], para_dict['dataset_name']+'_weight')))
    optimizer = optim.Adam(list(model.parameters()), lr=para_dict['lr'], weight_decay=0)
    for phase in phases:
        model.eval()
        epoch_loss, epoch_roc = single_linear(phase, dataloader, model, optimizer, para_dict)
        print('%.4f' % epoch_loss, '%.4f' % epoch_roc)
    return epoch_roc




def main():
    para_dict = {}
    para_dict['device'] = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    para_dict['batch_size'] = 128
    para_dict['embedding_dim'] = 128
    para_dict['shuffles'] = {'train': True, 'val': False, 'test': False}
    para_dict['max_epochs'] = 400
    index = 2
    max_seed = 5
    setup_seed(1)
    para_dict['basepath_to_data'] = os.path.join(os.path.abspath('..'), 'data')
    para_dict['lr'] = 1e-3
    para_dict['patience'] = 8
    frist_dataset_list = ['chapman', 'physionet2020', 'ptb_xl'][index:index + 1]
    second_dataset_list = [['physionet2020', 'ptb_xl'], ['chapman', 'ptb_xl'], ['chapman', 'physionet2020']][index:index + 1]
    frist_noutputs = [4, 9, 5][index:index + 1]
    second_noutputs_list = [[9, 5], [4, 5], [4, 9]][index:index + 1]
    model_list = ['AlexNet', 'VGG', 'ResNet'][2:]
    for first_dataset, frist_noutput, second_datasets, second_noutputs in zip(frist_dataset_list, frist_noutputs, second_dataset_list, second_noutputs_list):
        for model_name in model_list:
            para_dict['model_name'] = model_name
            for seed in range(max_seed):
                para_dict['dataset_name'] = first_dataset
                print(first_dataset, 'model_name=', para_dict['model_name'], 'seed: ', seed)
                para_dict['save_path_dir'] = os.path.join('Results', para_dict['dataset_name'], para_dict['model_name'], 'seed'+str(seed))
                if os.path.isdir(para_dict['save_path_dir']) == False:
                    os.makedirs(para_dict['save_path_dir'])

                # 无监督训练
                para_dict['noutputs'] = frist_noutput
                if seed != 0:
                    cl_train(para_dict)

                # 线性训练
                if para_dict['dataset_name'] != 'pdb':
                    linear_train(para_dict)
                    epoch_roc = linear_test(para_dict)

                for second_dataset, second_noutput in zip(second_datasets, second_noutputs):
                    para_dict['dataset_name'] = second_dataset
                    para_dict['noutputs'] = second_noutput
                    finetune_train(para_dict)
                    epoch_roc = finetune_test(para_dict)





if __name__ == "__main__":
    main()
