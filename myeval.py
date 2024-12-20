# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :myeval
# @Time      :2024/9/1 15:09
# @Author    :Chen
"""
import numpy as np

def main():
    data = np.load('Result_fra.npy', allow_pickle=True).item()
    roc_dict = {}
    roc_std_dict = {}
    for dataset in list(data.keys()):
        roc_dict[dataset] = []
        roc_std_dict[dataset] = []
        for noise_type in list(data[dataset].keys()):
            roc = np.array(data[dataset][noise_type])
            roc_mean = np.mean(roc, axis=1)
            roc_std = np.std(roc, axis=1)
            roc_dict[dataset].append(roc_mean)
            roc_std_dict[dataset].append(roc_std)
        roc_dict[dataset] = np.array(roc_dict[dataset]).T
        roc_std_dict[dataset] = np.array(roc_std_dict[dataset]).T
    print(data)
    for dataset in list(roc_dict.keys()):
        data = roc_dict[dataset]
        std = roc_std_dict[dataset]
        print(data)
        print(std)


if __name__ == "__main__":
    main()
