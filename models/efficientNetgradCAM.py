import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from dataset.gradCAMtorchData import torchData

device = torch.device("cpu")


# Utility Functions
def get_data_loader(
    subset: str, transforms=None, batch_size=16
) -> tuple[DataLoader, dict[int, str]]:
    """
    [TODO:description]

    :param subset: [TODO:description]
    :param transforms [TODO:type]: [TODO:description]
    :param batch_size [TODO:type]: [TODO:description]
    :return: [TODO:description]
    :raises ValueError: [TODO:description]
    """
    if subset not in ["Training", "Testing", "Validation"]:
        raise ValueError("Invalid Subset!!!")
    dataset = torchData(subset, transform=transforms)
    class_dict = dataset.classLabels
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), class_dict


def model_init() -> nn.Module:
    """
    [TODO:description]

    :return: [TODO:description]
    """
    weights = torch.load(
        "./trainedModels/EfficientNetParams169.pt",
        map_location=device,
        weights_only=True,
    )
    model = efficientnet_b2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=4, bias=True),
    )
    model.load_state_dict(weights)
    return model


class gradModel(nn.Module):
    """
    [TODO:description]
    """

    def __init__(self):
        super(gradModel, self).__init__()
        self.efficient_model = model_init()
        self.features_conv = self.efficient_model.features
        self.avg_pool = self.efficient_model.avgpool
        self.classifier = self.efficient_model.classifier
        self.gradients = None

    def gradient_hook(self, grad):
        """
        [TODO:description]

        :param grad [TODO:type]: [TODO:description]
        """
        self.gradients = grad

    def forward(self, x):
        """
        [TODO:description]

        :param x [TODO:type]: [TODO:description]
        """
        x = self.features_conv(x)
        h = x.register_hook(self.gradient_hook)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_activation_gradient(self):
        """
        [TODO:description]
        """
        return self.gradients

    def get_activations(self, x):
        """
        [TODO:description]

        :param x [TODO:type]: [TODO:description]
        """
        return self.features_conv(x)


def plot_heatmap(gradCAM, img, label):
    """
    [TODO:description]

    :param gradCAM [TODO:type]: [TODO:description]
    :param img [TODO:type]: [TODO:description]
    :param label [TODO:type]: [TODO:description]
    """
    output = gradCAM(img)
    predicted = output.argmax(dim=1)
    print(f"Predicted: {predicted.item()}, Actual: {label.item()}")
    y_values = (predicted.item(), label.item())
    output[:, predicted].backward()
    gradients = gradCAM.get_activation_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    pooled_gradients = pooled_gradients.view(1, -1, 1, 1)
    activations = gradCAM.get_activations(img).detach()
    activations *= pooled_gradients
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.clamp(heatmap, min=0)
    heatmap /= torch.max(heatmap)

    # plt.matshow(heatmap.numpy(), cmap="inferno")
    # plt.colorbar()
    # plt.show()
    return heatmap.numpy(), y_values


def plot_gradcam(model, img, label, img_path: str):
    """
    [TODO:description]

    :param model [TODO:type]: [TODO:description]
    :param img [TODO:type]: [TODO:description]
    :param label [TODO:type]: [TODO:description]
    :param img_path: [TODO:description]
    """
    heatmap, y_values = plot_heatmap(model, img, label)
    heatmap = cv2.resize(heatmap, (img.shape[-1], img.shape[-2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Read the original image using OpenCV
    original_image = cv2.imread(img_path)
    original_image = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    return superimposed_img, y_values


def plot_pred_actual(original_image, segmented_image, y_values, label_dict):
    plt.figure(figsize=(10, 10))
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(original_image)
    axarr[0].set_title(f"Original Image\nTumor Class:{label_dict[y_values[1]]}")
    axarr[1].imshow(segmented_image)
    axarr[1].set_title(f"gradCAM Image\nTumor Class: {label_dict[y_values[0]]}")
    plt.suptitle(
        "Comparison of GradCAM segementation vs Actual Tumor Location",
        x=(fig.subplotpars.right + fig.subplotpars.left) / 2,
    )
    plt.show()


if __name__ == "__main__":
    # Load test data
    test_loader, label_dict = get_data_loader(
        "Testing", transforms=EfficientNet_B2_Weights.DEFAULT.transforms(), batch_size=1
    )
    # Initialize Grad-CAM model
    gradCAM = gradModel()
    gradCAM.eval()
    gradCAM.to(device)
    # Get a sample image and label from the test loader
    img, label, path = next(iter(test_loader))
    path = path[0]
    # Ensure the image is sent to the same device as the model
    img = img.to(torch.device(device))  # Update if you're using GPU
    label = label.to(torch.device(device))
    original_image = cv2.imread(path)
    # Plot Grad-CAM
    segmented_image, y_values = plot_gradcam(gradCAM, img, label, path)
    segmented_image = cv2.resize(
        segmented_image, (original_image.shape[1], original_image.shape[0])
    )
    plot_pred_actual(original_image, segmented_image, y_values, label_dict)
