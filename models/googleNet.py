import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import GoogLeNet_Weights, googlenet

model = googlenet(GoogLeNet_Weights.DEFAULT)
print(model.parameters)
