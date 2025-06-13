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

            return inputs
        return inputs.to(device)