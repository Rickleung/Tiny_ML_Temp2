import sys
import zipfile
import pickle
import torch

sys.path.append('/home/rickleung/PycharmProjects/pythonProject/Tiny_ML_Temp/algorithm/core/utils')
sys.path.append('/home/rickleung/PycharmProjects/pythonProject/Tiny_ML_Temp/algorithm/quantize')

from config import configs
from custom_quantized_format import build_quantized_network_from_cfg
from quantize_helper import create_scaled_head, create_quantized_head

__all__ = ['build_mcu_model']


def build_mcu_model():
    cfg_path = "/home/rickleung/PycharmProjects/pythonProject/Tiny_ML_Temp/compilation/mcunet-5fps/archive/data.pkl"

    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)

    model = build_quantized_network_from_cfg(cfg, n_bit=8)

    if configs.net_config.mcu_head_type == 'quantized':
        model = create_quantized_head(model)
    elif configs.net_config.mcu_head_type == 'fp':
        model = create_scaled_head(model, norm_feat=False)
    else:
        raise NotImplementedError

    return model
