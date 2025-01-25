import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import GoogLeNet_Weights, googlenet


def model_init():
    google_model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
    data_transforms = GoogLeNet_Weights.DEFAULT.transforms()
    # Freezing the base layers of the model
    for param in google_model.parameters():
        param.requires_grad = False


if __name__ == "__main__":
    model = model_init()
