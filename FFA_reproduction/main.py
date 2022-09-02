'''
import pacakge
1 参数相关的包 argparse
2 日志相关的包 loging tensorboard
3 模型文件 torch 自定义的网络等
4 指标相关的包 torchmetrics
'''
import os, time
import argparse
import tensorboard
import torch
import torchmetrics
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
import math
import numpy as np
from .models.FFA import FFA
from .metrics_utils import ssim , psnr

'''
模型参数
'''
def ger_args_parser():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--steps', type=int, default=10000)  # 迭代次数
	parser.add_argument('--device', type=str, default='Automatic detection')  # 设备？
	parser.add_argument('--resume', type=bool, default=True)  # 接着以前的模型进行训练
	parser.add_argument('--eval_step', type=int, default=500)  # 计算一下阶段性的结果
	parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')  # 学习率
	parser.add_argument('--model_dir', type=str, default='./trained_models/')  #
	parser.add_argument('--trainset', type=str, default='its_train')
	parser.add_argument('--testset', type=str, default='its_test')
	parser.add_argument('--net', type=str, default='ffa')
	parser.add_argument('--gps', type=int, default=3, help='residual_groups')
	parser.add_argument('--blocks', type=int, default=1, help='residual_blocks')
	parser.add_argument('--bs', type=int, default=1, help='batch size')
	parser.add_argument('--crop', action='store_true')
	parser.add_argument('--crop_size', type=int, default=16, help='Takes effect when using --crop ')
	parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
	parser.add_argument('--perloss', action='store_true', help='perceptual loss')

	opt = parser.parse_args()

	return opt
'''
 日 志
'''

'''
模型部分
	学习率调整器
'''
models_={
	'ffa':FFA(gps=opt.gps,blocks=opt.blocks),#opt.blocks
}
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net, loader_train, loader_test, optim, criterion):
		losses = []
		start_step = 0
		max_ssim = 0
		max_psnr = 0
		ssims = []
		psnrs = []
		if opt.resume and os.path.exists(opt.model_dir):
			print(f'resume from {opt.model_dir}')
			ckp = torch.load(opt.model_dir)
			losses = ckp['losses']
			net.load_state_dict(ckp['model'])
			start_step = ckp['step']
			max_ssim = ckp['max_ssim']
			max_psnr = ckp['max_psnr']
			psnrs = ckp['psnrs']
			ssims = ckp['ssims']
			print(f'start_step:{start_step} start training ---')
		else:
			print('train from scratch *** ')
		for step in range(start_step + 1, opt.steps + 1):
			net.train()
			lr = opt.lr
			if not opt.no_lr_sche:
				lr = lr_schedule_cosdecay(step, T)
				for param_group in optim.param_groups:
					param_group["lr"] = lr
			x, y = next(iter(loader_train))
			x = x.to(opt.device);
			y = y.to(opt.device)
			out = net(x)
			loss = criterion[0](out, y)
			if opt.perloss:
				loss2 = criterion[1](out, y)
				loss = loss + 0.04 * loss2
			loss.backward()

			# loss.backward()
			optim.step()
			optim.zero_grad()
			losses.append(loss.item())
			print(
				f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
				end='', flush=True)

			# with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
			#	writer.add_scalar('data/loss',loss,step)

			if step % opt.eval_step == 0:
				with torch.no_grad():
					ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

				print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

				# with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
				# 	writer.add_scalar('data/ssim',ssim_eval,step)
				# 	writer.add_scalar('data/psnr',psnr_eval,step)
				# 	writer.add_scalars('group',{
				# 		'ssim':ssim_eval,
				# 		'psnr':psnr_eval,
				# 		'loss':loss
				# 	},step)
				ssims.append(ssim_eval)
				psnrs.append(psnr_eval)
				if ssim_eval > max_ssim and psnr_eval > max_psnr:
					max_ssim = max(max_ssim, ssim_eval)
					max_psnr = max(max_psnr, psnr_eval)
					torch.save({
						'step': step,
						'max_psnr': max_psnr,
						'max_ssim': max_ssim,
						'ssims': ssims,
						'psnrs': psnrs,
						'losses': losses,
						'model': net.state_dict()
					}, opt.model_dir)
					print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

		# np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
		# np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
		# np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	#s=True
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		pred=net(inputs)
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		# ssims.append(ssim1)
		# psnrs.append(psnr1)

	return np.mean(ssims) ,np.mean(psnrs)

if __name__ =="__main__":
	opt = ger_args_parser()

print(torch.cuda.get_device_capability())