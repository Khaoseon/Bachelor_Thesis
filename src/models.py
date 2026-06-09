import torch
import torch.nn as nn


def conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.pool(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)


class VGGSmallBN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(in_channels, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(2),
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)


def get_model(model_name: str, in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    normalized_name = model_name.lower()

    if normalized_name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if normalized_name == "vgg_small_bn":
        return VGGSmallBN(in_channels=in_channels, num_classes=num_classes)

    raise ValueError(
        f"Modele inconnu: {model_name!r}. "
        "Modeles disponibles: custom_cnn, vgg_small_bn."
    )


if __name__ == "__main__":
    model = CustomCNN()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Shape de sortie du modele: {output.shape}")
