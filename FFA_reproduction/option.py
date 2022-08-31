import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()

'''
argparse 模块可以让人轻松编写用户友好的命令行接口。
程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
'''

# 添加参数
#  default - 当参数未在命令行中出现时使用的值。
#  type - 命令行参数应当被转换成的类型。
#  action='store_true'，只要运行时该变量有传参就将该变量设为True
#  在 add_argument 前，给属性名之前加上“--”，就能将之变为可选参数。
parser.add_argument('--steps',type=int,default=10000)   #迭代次数
parser.add_argument('--device',type=str,default='Automatic detection') # 设备？
parser.add_argument('--resume',type=bool,default=True)   # 接着以前的模型进行训练
parser.add_argument('--eval_step',type=int,default=500)  # 计算一下阶段性的结果
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')  # 学习率
parser.add_argument('--model_dir',type=str,default='./trained_models/')    #
parser.add_argument('--trainset',type=str,default='its_train')
parser.add_argument('--testset',type=str,default='its_test')
parser.add_argument('--net',type=str,default='ffa')
parser.add_argument('--gps',type=int,default=3,help='residual_groups')
parser.add_argument('--blocks',type=int,default=1,help='residual_blocks')
parser.add_argument('--bs',type=int,default=1,help='batch size')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=16,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss')

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+str(opt.gps)+'_'+str(opt.blocks)
opt.model_dir=opt.model_dir+model_name+'.pk'
log_dir='logs/'+model_name

print(opt)

print('model_dir:',opt.model_dir)


if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
