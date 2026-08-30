import torch.nn as nn
from torchvision.models import resnet18


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Returns a torchvision ResNet-18 model adapted for CIFAR-10 sized inputs.
    """
    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = resnet18(weights=None, num_classes=num_classes)

    # CIFAR-10 images are 32x32, much smaller than ImageNet's 224x224.
    # The default ResNet-18 stem (7x7 conv + maxpool) downsamples too
    # aggressively for such small images, so we replace it with a
    # smaller conv and drop the initial maxpool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model