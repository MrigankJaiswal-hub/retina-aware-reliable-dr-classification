from typing import Optional

import torch.nn as nn
import torchvision.models as models


def build_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_p: Optional[float] = None,
) -> nn.Module:
    """
    Build classification model for diabetic retinopathy grading.

    Supported models:
    - efficientnet_b0
    - efficientnet_b3
    - resnet50
    - densenet121
    """
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        if dropout_p is None:
            dropout_p = 0.2
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        if dropout_p is None:
            dropout_p = 0.3
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            "Choose from: efficientnet_b0, efficientnet_b3, resnet50, densenet121"
        )

    return model