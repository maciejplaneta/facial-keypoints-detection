import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import features_columns

# 1st Model 

# Images have (96, 96, 1) shape, thats wht CH = 1
INPUT_CHANNELS = 1 
FILTERS_NUMBER = 10
CONV_KERNEL_SIZE = 3
MAX_POOL_KERNEL_SIZE = 3

class CnnModel(nn.Module):
    """Some Information about CnnModel"""
    def __init__(self):
        super(CnnModel, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=MAX_POOL_KERNEL_SIZE)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=MAX_POOL_KERNEL_SIZE)
        )

        # self.third_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
        #     nn.BatchNorm2d(num_features=FILTERS_NUMBER),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
        #     nn.BatchNorm2d(num_features=FILTERS_NUMBER),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=MAX_POOL_KERNEL_SIZE)
        # )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=FILTERS_NUMBER * 8 * 8, out_features=len(features_columns))
        )

    def forward(self, x):
        x = self.first_layer(x)
        # print(x.shape)
        x = self.second_layer(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x