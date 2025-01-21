# General Imports
import time
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch._dynamo import optimize
from torch.cuda import device_count
from torch.nn.modules.loss import _Loss
from torch.types import Tensor
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from dataset.torchData import torchData

# Pytorch Imports


# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Utility Functions
def getDataLoader(
    subset: str, transforms=None, batchSize: int = 16
) -> tuple[DataLoader, int]:
    """
    A simple dataLoader wrapper that retrieves the subset of the original dataset
    and returns a dataLoader object with a user-defined batch size.

    :param subset: The subset to be loaded
    :param batchSize: Determines how many images should be loaded into each batch.
    :return: A dataLoader object that can be iterated through when training the
             model.
    """
    dataSet = torchData(subset, transform=transforms)
    return DataLoader(dataSet, batch_size=batchSize, shuffle=True), len(dataSet)


def initWeights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def modelInit() -> tuple:
    """
    Initializes the EfficientNet_B2 model with the corresponding weights.

    :return: A torch.nn.Module neural network with pretrained weights.
    """
    efficientModel = efficientnet_b2(EfficientNet_B2_Weights.DEFAULT)
    efficientTransforms = EfficientNet_B2_Weights.DEFAULT.transforms()
    # Sending the intialized model to the specified Processing Unit.
    efficientModel.to(device)
    # Freezing the base layers of the pretrained model to prevent updating
    # weights during training
    for param in efficientModel.parameters():
        param.requires_grad = False
    # Modifying the output layer to have 4 outputs instead of the 1000 output neurons.
    efficientModel.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=4, bias=True),
    )
    # Unfreezing the last 2 layers
    for param in efficientModel.features[-2:].parameters():
        param.requires_grad = True
    for param in efficientModel.classifier.parameters():
        param.requires_grad = True
    # Initializing the weights of the new layer using the Xavier Uniform Distribution.
    efficientModel.apply(initWeights)
    print(efficientTransforms)
    return efficientModel, efficientTransforms


def fitModel(
    model: nn.Module,
    batchDataSet: dict[str, tuple[DataLoader, int]],
    optimizer: optim.Optimizer,
    criterion: _Loss,
    numIters: int = 15,
) -> nn.Module:
    """
    The training loop for the EfficientNet_B2 model that utilizes user-defined optimizer
    and loss criterion to perform weight updates of the outer layer and provide statistics.

    :param model: A nn.Module neural network that has defined layers and forward propogation
    defined.
    :param batchDataSet: Dictionary of all three subsets along with the numeber of images.
    :param optimizer: The optimizer to use for updating the weights.
    :param criterion: The error function that returns the loss of the model.
    :param numIters: The number of epochs to train the model for.
    :return: A trained model with updated weights.
    """
    model.cuda()
    bestAcc = 0.0
    startTime = time.time()
    for epoch in range(numIters):
        epochTime = time.time()
        for phase in ["Training", "Validating"]:
            print(f"\nStarting Epoch: {epoch}/{numIters}\nPhase: {phase}\n")
            # Setting to Training Mode
            model.train(phase == "Training")
            # Initializing statistics variables
            runningLoss = torch.tensor(0.0, device=device)
            runningCorrect = torch.tensor(0, device=device)
            for images, labels in batchDataSet[phase][0]:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # Track predictions if in training phase otherwise just make predictions.
                with torch.set_grad_enabled(phase == "Training"):
                    classProbabilities = model(images)
                    classProbabilities.to(device)
                    yHats = torch.argmax(
                        torch.softmax(classProbabilities, dim=1), dim=1
                    )
                    loss = criterion(classProbabilities, labels)
                    if phase == "Training":
                        loss.backward()
                        optimizer.step()

                runningLoss += loss.item() * images.size(0)
                runningCorrect += torch.sum(yHats == labels)
            epochLoss = runningLoss / batchDataSet[phase][1]
            epochAcc = runningCorrect / batchDataSet[phase][1]

            if phase == "Validating" and epochAcc > bestAcc:
                bestAcc = epochAcc
                torch.save(model.state_dict(), f"./EfficientNetParams{epoch}.pt")
            print(
                f"""{'='*30}
                  Finished Epoch: {epoch}/{numIters}
                  Time Taken: {(time.time()-epochTime)/60:.2f} Minutes
                  Current Loss: {epochLoss:.4f}
                  Current Accuracy: {epochAcc*100:.2f}%
                  {'='*30}"""
            )

    print(f"Total Time Taken: {(time.time() - startTime)/60:.2f} Minutes")
    return model


if __name__ == "__main__":
    model, transforms = modelInit()
    dataLoaders = {
        "Training": getDataLoader("Training", transforms),
        "Testing": getDataLoader("Testing", transforms),
        "Validating": getDataLoader("Validation", transforms),
    }
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    fitModel(model, dataLoaders, optimizer, criterion)
