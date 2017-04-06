import tensorflow as tf
import os.path as osp
import os

op_file = 'roi_pooling_op_gpu_cuda8.so'  # for CUDA 8
#op_file = 'roi_pooling_op_gpu.so' # CUDA 7.5

filename = osp.join(osp.dirname(__file__), op_file)

_roi_pooling_module = tf.load_op_library(filename)
roi_pool = _roi_pooling_module.roi_pool
roi_pool_grad = _roi_pooling_module.roi_pool_grad
