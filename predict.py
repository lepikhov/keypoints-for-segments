import argparse

import config
import cv2
import numpy as np
import torch
from model import HorseKeypointResNet50

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--segment', help='must be in (head, neck, body, front leg, rear leg, horse)')
parser.add_argument('-i', '--image', help='must be in (0..907)')
args = vars(parser.parse_args())

segment = args['segment']
image_number = args['image']

match segment:
    case 'head': keypoints_number = len(config.HEAD_KEYPOINTS)
    case 'neck': keypoints_number = len(config.NECK_KEYPOINTS)
    case 'body': keypoints_number = len(config.BODY_KEYPOINTS)
    case 'front leg': keypoints_number = len(config.FRONTLEG_KEYPOINTS)
    case 'rear leg': keypoints_number = len(config.REARLEG_KEYPOINTS)
    case 'horse': keypoints_number = len(config.HORSE_KEYPOINTS)    
    case _:
        pass

model = HorseKeypointResNet50(
    pretrained=False, requires_grad=False, keypoints_number=keypoints_number).to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load(f"{config.ROOT_OUTPUT_DIRECTORY}/{segment}/model.pth")
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
print('best epoch =', checkpoint['epoch'])
model.eval()

with torch.no_grad():
    image = cv2.imread(f"{config.ROOT_OUTPUT_DIRECTORY}/{segment}/{image_number}.jpg")
    orig_image = image.copy()
    orig_h, orig_w, c = orig_image.shape  
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0).to(config.DEVICE)
    outputs = model(image)
outputs = outputs.cpu().detach().numpy()
outputs = outputs.reshape(-1, 3)
keypoints = outputs

for p in range(keypoints.shape[0]):
    cv2.circle(orig_image, (int(keypoints[p, 0]*orig_w/224), orig_h-int(keypoints[p, 1]*orig_h/224)),
               int(orig_h/200)+1, (0, 0, 255), -1, cv2.LINE_AA)

cv2.imwrite(f"{config.ROOT_OUTPUT_DIRECTORY}/predicted_{segment}_{image_number}.png", orig_image)
