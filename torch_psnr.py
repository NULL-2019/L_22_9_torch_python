import torch
import numpy as np
import torchmetrics
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image

psnr = PeakSignalNoiseRatio()

preds = Image.open(r"C:\MyDataset\test_dataset\ITS\gt\1400.png")
target = Image.open(r"C:\MyDataset\test_dataset\ITS\haze\1400_1.png")
preds = np.array(preds)
target = np.array(target)
preds = torch.tensor(preds)
target = torch.tensor(target)
mypsnr = psnr(target,preds)
print(mypsnr)