import torchmetrics
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
psnr = PeakSignalNoiseRatio()
ssim = StructuralSimilarityIndexMeasure()