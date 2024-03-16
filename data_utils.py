import pandas as pd
import numpy as np
import torch

features_columns = [
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x','right_eye_center_y',
            'left_eye_inner_corner_x','left_eye_inner_corner_y',
            'left_eye_outer_corner_x','left_eye_outer_corner_y',
            'right_eye_inner_corner_x','right_eye_inner_corner_y',
            'right_eye_outer_corner_x','right_eye_outer_corner_y',
            'left_eyebrow_inner_end_x','left_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x','left_eyebrow_outer_end_y',
            'right_eyebrow_inner_end_x','right_eyebrow_inner_end_y',
            'right_eyebrow_outer_end_x','right_eyebrow_outer_end_y',
            'nose_tip_x','nose_tip_y',
            'mouth_left_corner_x','mouth_left_corner_y',
            'mouth_right_corner_x','mouth_right_corner_y',
            'mouth_center_top_lip_x','mouth_center_top_lip_y',
            'mouth_center_bottom_lip_x','mouth_center_bottom_lip_y'
            ]

# Function to convert each row into a NumPy array
def row_to_np(row):
    return np.array(row.split()).reshape(96, 96).astype(float)


# Function to calculate mean and standard deviation for given dataloader
def mean_std(loader):
    mean, std = 0, 0
    for images, _ in loader:
        # shape of images = [b,c,w,h]
        mean += images.mean([0,2,3])
        std += images.std([0,2,3])
    mean = mean/loader.batch_size
    std = std/loader.batch_size
    print(f'mean: {mean}, std: {std}')
    return mean, std


class FacialKeypointsDataset(torch.utils.data.Dataset):
    """Some Information about FacialKeypointsDataset"""
    def __init__(self, data_path, transform=None, train=True):
        df = pd.read_csv(data_path)
        self.train = train
        self.features_columns = features_columns
        self.transform = transform

        if train:
            self.features = df[self.features_columns].ffill()
        self.images = df['Image'].apply(row_to_np)

    def __getitem__(self, index):

        if self.train:
            features = torch.tensor(self.features.iloc[index].to_numpy()).type(torch.float32)
            image = torch.tensor(self.images.iloc[index]).unsqueeze(dim=0).type(torch.float32)

            if self.transform is not None:
                image = self.transform(image)

            return image, features
        else:
            return torch.tensor(self.images.iloc[index]).unsqueeze(dim=0).type(torch.float32)

    def __len__(self):
        return len(self.images)
