import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCIFAR100Model_torch(nn.Module):

    def __init__(self):
        super(SimpleCIFAR100Model_torch, self).__init__()
        
        # 1st conv block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # Layer 1
        self.bn1 = nn.BatchNorm2d(64)                                                     # Layer 2
        
        # 2nd conv block
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)                                    # Layer 3
        self.bn2 = nn.BatchNorm2d(128)                                                   # Layer 4
        
        # 3rd conv block
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)                                   # Layer 5
        self.bn3 = nn.BatchNorm2d(256)                                                   # Layer 6
        
        # Fully connected layers
        # After 3 max pools, 32x32 -> 16x16 -> 8x8 -> 4x4 spatial size
        self.fc1 = nn.Linear(256 * 4 * 4, 512)                                           # Layer 7
        self.fc2 = nn.Linear(512, 100)                                                   # Layer 8
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        
        x = x.view(-1, 256 * 4 * 4)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Note: no softmax here, outputs logits
        
        return x
