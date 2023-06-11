import sys
sys.path.append('/home/rickleung/PycharmProjects/pythonProject/Tiny_ML_Temp/compilation/convert')

from load_mcunetv3 import (
    build_quantized_mcunet,
    build_quantized_mbv2,
    build_quantized_proxyless,
)

from pth2ir import (
    pth_model_to_ir,
    generated_backward_graph,
)
