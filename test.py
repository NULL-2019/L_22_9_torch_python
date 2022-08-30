import os
import torch

torch.device("cuda")
print(torch.cuda.is_available())

print(os.getcwd())