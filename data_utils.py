# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :data_utils.py
# @Time      :2024/4/12 12:04
# @Author    :Chen
"""
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from operator import itemgetter
import os
import pickle
import numpy as np
#import os
import random
from scipy.signal import resample



class my_dataset_contrastive(Dataset):
    def __init__(self, phase, path, dataset, fra=1):
        self.data_path = path
        self.dataset_name = dataset
        self.phase = phase
        self.fra = fra
        input_array, output_array = self.load_raw_inputs_and_outputs()
        input_array, output_array, pid_array = self.retrieve_classifal_data(input_array, output_array)
        self.input_array = input_array
        self.label_array = output_array
        self.pids = pid_array
        self.phase = phase


    def load_raw_inputs_and_outputs(self):
        if self.fra < 1 and self.phase == 'train':
            with open(os.path.join(self.data_path, self.dataset_name, 'frames_phases_train_%s.pkl' % (str(self.fra))), 'rb') as f:
                data = pickle.load(f)
        else:
            with open(os.path.join(self.data_path, self.dataset_name, 'frames_phases_%s.pkl' % (self.phase)),
                      'rb') as f:
                data = pickle.load(f)

        return data

    def retrieve_classifal_data(self, input_array, output_array):
        pid_list = list(input_array)
        inputs = []
        outputs = []
        pids = []
        for pid in pid_list:
            inputs.append(input_array[pid])
            outputs.append(output_array[pid])
            if self.dataset_name in ['chapman', 'pdb']:
                pids.append(pid)
            elif self.dataset_name in ['physionet2020', 'ptb_xl']:
                pids.append(pid.split('_')[0])
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        pids = np.array(pids)
        return inputs, outputs, pids


    def normalize_frame(self,frame):
        if isinstance(frame,np.ndarray):
            frame = (frame - np.min(frame))/(np.max(frame) - np.min(frame) + 1e-8)
        elif isinstance(frame, torch.Tensor):
            frame = (frame - torch.min(frame))/(torch.max(frame) - torch.min(frame) + 1e-8)
        return frame


    def __getitem__(self, index):
        input_frame = self.input_array[index]
        label = self.label_array[index]
        pid = self.pids[index]
        label = torch.tensor(label, dtype=torch.float)

        if input_frame.dtype != float:
            input_frame = np.array(input_frame, dtype=float)
        frame = torch.tensor(input_frame, dtype=torch.float).unsqueeze(0)
        frame_views = torch.empty(1, 2048, 24)
        n = 0
        for lead in range(frame.shape[1]):
            start = 0
            for i in range(2):
                current_view = frame[0, lead, start:start + 2048]
                current_view = self.normalize_frame(current_view)
                frame_views[0, :, n] = current_view
                start += 2048
                n += 1

        return frame_views, label, pid

    def __len__(self):
        return len(self.input_array)

def obtain_contrastive_loss(latent_embeddings):
    loss = 0.0
    n = 0
    for i in range(12):
        view1_array = latent_embeddings[:, :, 2*i]  # (BxH)
        view2_array = latent_embeddings[:, :, 2*i+1]  # (BxH)
        norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
        norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
        sim_matrix = torch.mm(view1_array, view2_array.transpose(0, 1))
        norm_matrix = torch.mm(norm1_vector.transpose(0, 1), norm2_vector)
        temperature = 0.1
        argument = sim_matrix / (norm_matrix * temperature)
        sim_matrix_exp = torch.exp(argument)
        diag_elements = torch.diag(sim_matrix_exp)
        triu_sum = torch.sum(sim_matrix_exp, 1)
        tril_sum = torch.sum(sim_matrix_exp, 0)

        loss += -torch.mean(torch.log(diag_elements / (triu_sum)))
        loss += -torch.mean(torch.log(diag_elements / (tril_sum)))

        n += 2
    loss = loss/n
    return loss



def obtain_lead_loss(inputs, latent_embeddings):
    loss = 0.0
    n = 0
    inputs = inputs.squeeze(dim=1)
    for patient in range(inputs.shape[0]):
        input_data = inputs[patient, :, :].transpose(0, 1)
        latent_data = latent_embeddings[patient, :, :].transpose(0, 1)
        for i in range(2):
            input_array = input_data[i::2, :]
            latent_array = latent_data[i::2, :]
            input_norm_vector = input_array.norm(dim=1).unsqueeze(0)
            input_sim_matrix = torch.mm(input_array, input_array.transpose(0, 1))
            input_norm_matrix = torch.mm(input_norm_vector.transpose(0, 1), input_norm_vector)
            input_matrix = input_sim_matrix / (input_norm_matrix + 1e-8)
            latent_norm_vector = latent_array.norm(dim=1).unsqueeze(0)
            latent_sim_matrix = torch.mm(latent_array, latent_array.transpose(0, 1))
            latent_norm_matrix = torch.mm(latent_norm_vector.transpose(0, 1), latent_norm_vector)
            latent_matrix = latent_sim_matrix / (latent_norm_matrix + 1e-8)
            input_distance = torch.reshape(input_matrix, (-1,))
            latent_distance = torch.reshape(latent_matrix, (-1,))
            loss += torch.nn.functional.mse_loss(input_distance, latent_distance)
        n += 1
    loss = loss/n
    return loss




