import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    """Concat along the channel dimension of AdaptiveAvgPool2d and AdaptiveMaxPool2d.

    Args:
        size: size of pooling, default: (1, 1)
    """

    def __init__(self, size=(1, 1)):
        super().__init__()
        self.size = size
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat((self.ap(x), self.mp(x)), dim=1)


class Flatten(nn.Module):
    """Flatten the input, but keep the batch-size."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def create_features(model, split):
    """Split the model and add AdaptiveConcatPool2d, Flatten layers.

    Args:
        model: base model
        split: index to split

    Returns:
        container: nn.Sequential container, of convolutional layers
    """
    conv_layers = list(model.children())[:split]
    flat_layers = [AdaptiveConcatPool2d(), Flatten()]
    features = conv_layers + flat_layers

    return nn.Sequential(*features)


def create_classifier(xtra_fc=[512], dropout=[0.78], classes=2):
    """Create classifier layer of the model.

    Args:
        xtra_fc: list of linear units
        dropout: dropout in each layer
        classes: number of output classes

    Returns:
        container: nn.Sequential container, of fully connected layers
    """
    fc = list()
    in_dim = 4096  # Compute dynamically
    for i, dim in enumerate(xtra_fc):
        fc += [
            nn.BatchNorm1d(in_dim),
            nn.Dropout(dropout[i]),
            nn.Linear(in_dim, dim),
            nn.ReLU(inplace=True)
        ]
        in_dim = dim

    fc += [
        nn.BatchNorm1d(in_dim),
        nn.Dropout(p=0.78),
        nn.Linear(in_dim, classes),
        nn.LogSoftmax(dim=1)
    ]

    return nn.Sequential(*fc)


class ResnetFinetuned(nn.Module):
    """Finetune Resnet model by adding custom head."""

    def __init__(self, model, split=8, xtra_fc=[512], dropout=[0.78], classes=2):
        super().__init__()
        self.features = create_features(model, split)
        self.classifier = create_classifier(xtra_fc, dropout, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
