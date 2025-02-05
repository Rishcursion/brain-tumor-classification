# General Imports import time
import time
from typing import Any, Literal

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchmetrics.classification import (Accuracy, F1Score, MulticlassAUROC,
                                         Precision, Specificity)
from torchmetrics.collections import MetricCollection
from torchmetrics.wrappers import MetricTracker
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from dataset.torchData import torchData as torch_data

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Utility Functions
def get_data_loader(
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
    data_set = torch_data(subset, transform=transforms)
    return DataLoader(data_set, batch_size=batchSize, shuffle=True), len(data_set)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def model_init() -> tuple:
    """
    Initializes the EfficientNet_B2 model with the corresponding weights.

    :return: A torch.nn.Module neural network with pretrained weights.
    """
    efficient_model = efficientnet_b2(EfficientNet_B2_Weights.DEFAULT)
    efficient_transforms = EfficientNet_B2_Weights.DEFAULT.transforms()
    # Sending the intialized model to the specified Processing Unit.
    efficient_model.to(device)
    # Freezing the base layers of the pretrained model to prevent updating
    # weights during training
    for param in efficient_model.parameters():
        param.requires_grad = False
    # Modifying the output layer Gto have 4 outputs instead of the 1000 output neurons.
    efficient_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=4, bias=True),
    )
    # Unfreezing the last 2 layers
    for param in efficient_model.features[-2:].parameters():
        param.requires_grad = True
    for param in efficient_model.classifier.parameters():
        param.requires_grad = True
    # Initializing the weights of the new layer using the Xavier Uniform Distribution.
    efficient_model.apply(init_weights)
    print(efficient_transforms)
    return efficient_model, efficient_transforms


# Training Function
def fit_model(
    model: nn.Module,
    batch_data_set: dict[str, tuple[DataLoader, int]],
    optimizer: optim.Optimizer,
    criterion: _Loss,
    num_iters: int = 15,
) -> tuple[nn.Module, dict[str, dict[str, Any]]]:
    """
    The training loop for the EfficientNet_B2 model that utilizes user-defined optimizer
    and loss criterion to perform weight updates of the outer layer and provide statistics.

    :param model: A nn.Module neural network that has defined layers and forward propogation
    defined.
    :param batch_data_set: Dictionary of all three subsets along with the numeber of images.
    :param optimizer: The optimizer to use for updating the weights.
    :param criterion: The error function that returns the loss of the model.
    :param num_iters: The number of epochs to train the model for.
    :return: A trained model with updated weights.
    """
    model.cuda()
    bestAcc = 0.0
    startTime = time.time()
    Metrics = {
        "Training": {"Loss": [], "Metrics": []},
        "Validating": {"Loss": [], "Metrics": []},
    }
    tracker = MetricTracker(
        MetricCollection(
            [
                Accuracy(task="multiclass", num_classes=4),
                F1Score(task="multiclass", num_classes=4),
                MulticlassAUROC(num_classes=4),
                Specificity(task="multiclass", num_classes=4),
                Precision(task="multiclass", num_classes=4),
            ]
        )
    )
    tracker.to(device)
    for epoch in range(num_iters):
        tracker.increment()
        epoch_time = time.time()
        for phase in ["Training", "Validating"]:
            print(f"\nStarting Epoch: {epoch+1}/{num_iters}\nPhase: {phase}\n")
            # Setting to Training Mode
            model.train(phase == "Training")
            # Initializing statistics variables
            running_loss = torch.tensor(0.0, device=device)
            running_correct = torch.tensor(0, device=device)
            # Start batch training
            for images, labels in batch_data_set[phase][0]:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # Track predictions if in training phase otherwise just make predictions.
                with torch.set_grad_enabled(phase == "Training"):
                    class_probabilities = model(images)
                    class_probabilities.to(device)
                    yHats = torch.argmax(
                        torch.softmax(class_probabilities, dim=1), dim=1
                    )
                    loss = criterion(class_probabilities, labels)
                    if phase == "Training":
                        loss.backward()
                        optimizer.step()
                tracker.update(class_probabilities, labels)
                running_loss += loss.item() * images.size(0)
                running_correct += torch.sum(yHats == labels)
            epoch_loss = running_loss / batch_data_set[phase][1]
            epoch_accuracy = running_correct / batch_data_set[phase][1]
            Metrics[phase]["Loss"].append(epoch_loss)
            Metrics[phase]["Metrics"].append(tracker.compute_all())
            if phase == "Validating" and epoch_accuracy > bestAcc:
                bestAcc = epoch_accuracy
                torch.save(model.state_dict(), f"./EfficientNetParams{epoch}.pt")
            print(
                f"""{'='*30}
                  Finished Epoch: {epoch+1}/{num_iters}
                  Time Taken: {(time.time()-epoch_time)/60:.2f} Minutes
                  Current Loss: {epoch_loss:.4f}
                  Current Accuracy: {epoch_accuracy*100:.2f}%
                  {'='*30}"""
            )

    print(f"Total Time Taken: {(time.time() - startTime)/60:.2f} Minutes")
    return model, Metrics


def write_stats(subset: dict, label: Literal["train", "val"]) -> None:
    with open(f"./statistics/efficient_net_{label}.csv", "a") as fp:
        headers = "Epoch, Loss, Accuracy"
        for i in subset.keys():
            headers += ", " + i
        fp.write(headers + "\n")


if __name__ == "__main__":
    model, transforms = model_init()
    data_loaders = {
        "Training": get_data_loader("Training", transforms),
        "Testing": get_data_loader("Testing", transforms),
        "Validating": get_data_loader("Validation", transforms),
    }
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    epochs = int(input("Enter The Number Of Epochs: "))
    model, Metrics = fit_model(model, data_loaders, optimizer, criterion, epochs)
    trainMetrics, testMetrics = Metrics["Training"], Metrics["Validating"]
    with open("./statistics/efficient_net_train.json","a") as fp:
        fp.write(f"{trainMetrics}")
    with open("./statistics/efficient_net_val.json", "a") as fp:
        fp.write("testMetrics")
