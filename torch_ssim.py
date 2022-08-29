import torch
import torchmetrics
from PIL import Image
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure()

preds = Image.open(r"C:\MyDataset\test_dataset\ITS\gt\1400.png")
target = Image.open(r"C:\MyDataset\test_dataset\ITS\haze\1400_1.png")
preds = np.array(preds)
target = np.array(target)
preds = torch.tensor(preds)
target = torch.tensor(target)

target = torch.tensor(target)
myssim = ssim(preds,target)