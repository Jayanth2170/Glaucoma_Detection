# fpd_student_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# TransScan Module (Adaptive receptive field adjustment)
class TransScan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransScan, self).__init__()
        # Two convolutions with different receptive fields (dilation and padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=32)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, groups=32)

        # Attention mechanism to focus on important features
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        combined = x1 + x2
        attention_map = self.attention(combined)
        return combined * attention_map


# Student model with FPD using the TransScan module
class StudentModelFPD(nn.Module):
    def __init__(self):
        super(StudentModelFPD, self).__init__()
        resnet = models.resnet50(pretrained=True)  # Using a pretrained ResNet50 as the feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Use all layers except the last two

        # Feature Projection and TransScan module
        self.trans_scan = TransScan(2048, 384)  # Update the input channels to 2048

        # Final fully connected layer for classification (binary classification for glaucoma)
        self.fc = nn.Linear(384, 2)

    def forward(self, x):
        features = self.feature_extractor(x)  # Extract features from input
        projected_features = self.trans_scan(features)  # Pass features through TransScan
        flattened = F.adaptive_avg_pool2d(projected_features, (1, 1)).view(x.size(0), -1)
        return self.fc(flattened)  # Final classification
