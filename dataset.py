import os
import random

import albumentations as A
import config
import cv2
import numpy as np
import pandas as pd
import torch
import utils
from sklearn import model_selection
from torch.utils.data import DataLoader, Dataset


def train_test_split(json_path, split):
    #df_data = pd.read_csv(csv_path)
    df_data = pd.read_json(os.path.join(
        config.ROOT_OUTPUT_DIRECTORY, json_path), orient='table')
    #len_data = len(df_data)
    # calculate the validation data sample length
    #valid_split = int(len_data * split)
    # calculate the training data samples length
    #train_split = int(len_data - valid_split)
    #training_samples = df_data.iloc[:train_split][:]
    #valid_samples = df_data.iloc[-valid_split:][:]

    training_samples, valid_samples = model_selection.train_test_split(df_data, shuffle=True,
                                                                       random_state=8,
                                                                       test_size=split)
    return training_samples, valid_samples


class HorseKeypointDataset(Dataset):
    def __init__(self, samples, path, transform=None):
        self.data = samples
        self.path = path
        self.resize = 224
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = cv2.imread(f"{self.path}/{self.data.iloc[index][0]}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get the keypoints
        keypoints = self.data.iloc[index][1]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 3)

        if self.transform:
            kps = []
            for i in range(keypoints.shape[0]):
                kps.append((keypoints[i][0], keypoints[i][1]))
            sample = self.transform(image=image, keypoints=kps)

            image = sample['image']
            kps = sample['keypoints']

            for i in range(keypoints.shape[0]):
                keypoints[i][0] = kps[i][0]
                keypoints[i][1] = kps[i][1]

        orig_h, orig_w, _ = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))
        # rescale keypoints according to image resize
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h, 1]

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }


def prepare_samples(segment):

    img_path = f"{config.ROOT_OUTPUT_DIRECTORY}/{segment}"
    s = segment.replace(" ", "")
    json_path = f"{s}_df.json"

    # transform for augmentation

    random.seed(7)

    transform = A.Compose([
        A.Rotate(limit=10, p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.OneOf([
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.7)
        ], p=1),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))

    training_samples, valid_samples = train_test_split(json_path,
                                                       config.TEST_SPLIT)

    # initialize the dataset - `HorseKeypointDataset()`
    train_data = HorseKeypointDataset(
        training_samples, img_path, transform=transform)

    valid_data = HorseKeypointDataset(valid_samples, img_path)
    # prepare data loaders
    train_loader = DataLoader(train_data,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True)
    valid_loader = DataLoader(valid_data,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False)

    print(f"Segment: {segment}")
    print(f"Training sample instances: {len(train_data)}")
    print(f"Validation sample instances: {len(valid_data)}")

    return train_data, valid_data, train_loader, valid_loader


if __name__ == "__main__":
    train_data, valid_data, train_loader, valid_loader = prepare_samples(
        "rear leg")

    im, kp = train_data.__getitem__(0)
