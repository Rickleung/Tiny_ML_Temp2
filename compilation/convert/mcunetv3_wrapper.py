import sys
sys.path.append('/home/rick/tiny-training/algorithm/core/model')
sys.path.append('/home/rick/tiny-training/algorithm/core/utils')
sys.path.append('/home/rick/tiny-training/algorithm/quantize')

from model_entry import build_mcu_model
from config import (
    configs,
    load_config_from_file,
    update_config_from_args,
    update_config_from_unknown_args,
)
from quantized_ops_diff import (
    QuantizedConv2dDiff,
    QuantizedMbBlockDiff,
    ScaledLinear,
    QuantizedAvgPoolDiff,
)
