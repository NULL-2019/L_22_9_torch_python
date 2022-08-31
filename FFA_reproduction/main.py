'''
import pacakge
1 参数相关的包 argparse
2 日志相关的包 loging tensorboard
3 模型文件 torch 自定义的网络等
4 指标相关的包 torchmetrics
'''
import argparse
import tensorboard
import torch
import torchmetrics
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
'''
模型参数
'''

'''
 日 志
'''
print(torch.cuda.get_device_capability())