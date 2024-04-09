import glob
import importlib
import yaml
import os
import torch
import numpy


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):        
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or isinstance(inputs, numpy.int64):  
                # AttributeError: 'numpy.int64' object has no attribute 'to'
                # sizhewei @ 2022/10/04
                # 添加类型 numpy.int64 数据集中的 sample_idx 是此类型
            return inputs
        return inputs.to(device)