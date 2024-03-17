import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import features_columns

# Images have (96, 96, 1) shape, thats wht CH = 1
INPUT_CHANNELS = 1 
FILTERS_NUMBER = 10
DROPOUT_RATE = 0.5
CONV_KERNEL_SIZE = 3
MAX_POOL_KERNEL_SIZE = 3


class TwoConvLayersModel(nn.Module):
    def __init__(self):
        super(TwoConvLayersModel, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

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


class TwoConvLayersWithDropoutsModel(nn.Module):
    def __init__(self):
        super(TwoConvLayersWithDropoutsModel, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.MaxPool2d(kernel_size=3)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.MaxPool2d(kernel_size=3)
        )

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


class ThreeConvLayersModel(nn.Module):
    def __init__(self):
        super(ThreeConvLayersModel, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.third_layer = nn.Sequential(
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=FILTERS_NUMBER * 8 * 8, out_features=len(features_columns))
        )

    def forward(self, x):
        x = self.first_layer(x)
        # print(x.shape)
        x = self.second_layer(x)
        # print(x.shape)
        x = self.third_layer(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x


class ThreeConvLayersWithDropoutsModel(nn.Module):
    def __init__(self):
        super(ThreeConvLayersWithDropoutsModel, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.MaxPool2d(kernel_size=2)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.MaxPool2d(kernel_size=2)
        )

        self.third_layer = nn.Sequential(
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(in_channels=FILTERS_NUMBER, out_channels=FILTERS_NUMBER, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm2d(num_features=FILTERS_NUMBER),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=FILTERS_NUMBER * 8 * 8, out_features=len(features_columns))
        )

    def forward(self, x):
        x = self.first_layer(x)
        # print(x.shape)
        x = self.second_layer(x)
        # print(x.shape)
        x = self.third_layer(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x