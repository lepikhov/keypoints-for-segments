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
    df_data = pd.read_json(os.path.join(
        config.ROOT_OUTPUT_DIRECTORY, json_path), orient='table')


    training_samples, valid_samples = model_selection.train_test_split(df_data, shuffle=True,
                                                                       random_state=None,
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
        h, _, _ = image.shape

        # get the keypoints
        keypoints = self.data.iloc[index][1]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 3)

        if self.transform:
            kps = []
            for i in range(keypoints.shape[0]):
                kps.append((keypoints[i][0], h-keypoints[i][1]))
            sample = self.transform(image=image, keypoints=kps)

            image = sample['image']
            kps = sample['keypoints']

            for i in range(keypoints.shape[0]):
                keypoints[i][0] = kps[i][0]
                keypoints[i][1] = h-kps[i][1]

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


def test_transformer(transformer, path, df, idx, counter):

    image = cv2.imread(f"{path}/{df.iloc[idx][0]}.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, _, _ = image.shape

    # get the keypoints
    keypoints = df.iloc[idx][1]
    keypoints = np.array(keypoints, dtype='float32')
    # reshape the keypoints
    keypoints = keypoints.reshape(-1, 3)

    kps = []
    for i in range(keypoints.shape[0]):
        kps.append((keypoints[i][0], h-keypoints[i][1]))

    sample = transformer(image=image, keypoints=kps)

    image = sample['image']
    kps = sample['keypoints']

    for i in range(keypoints.shape[0]):
        keypoints[i][0] = kps[i][0]
        keypoints[i][1] = h-kps[i][1]

    image = utils.draw_keypoints(keypoints, image)
    cv2.imwrite(f"outputs/transform/{idx}-{counter}.jpg", image)


def prepare_samples(segment, test_aug=False):

    img_path = f"{config.ROOT_OUTPUT_DIRECTORY}/{segment}"
    s = segment.replace(" ", "")
    json_path = f"{s}_df.json"

    # transform for augmentation

    #random.seed(42)

    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        #A.OneOf([
        #    A.HueSaturationValue(p=0.5),
        #    A.RGBShift(p=0.7),           
        #], p=0.5),
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
            A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1),     
            A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.6, alpha_coef=0.1, p=1),    
        ], p=0.25),
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
    
    if test_aug:
        for counter in range(20):
            test_transformer(transformer=transform, path=img_path, df=training_samples, idx=10, counter=counter)

    print(f"Segment: {segment}")
    print(f"Training sample instances: {len(train_data)}")
    print(f"Validation sample instances: {len(valid_data)}")

    return train_data, valid_data, train_loader, valid_loader


if __name__ == "__main__":
    train_data, valid_data, train_loader, valid_loader = prepare_samples("rear leg", test_aug=True)

    im, kp = train_data.__getitem__(0)
