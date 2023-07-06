import torch

ROOT_DATA_DIRECTORY ="/home/pavel/projects/horses/soft/python/morphometry/datasets/2023/segmentation" #path to dataset
ROOT_OUTPUT_DIRECTORY ="/home/pavel/projects/horses/soft/python/morphometry/keypoints-for-segments/outputs" 

SEGMENTS_NUMBER = 6

RECTANGLES_ATTRIBUTES = [
    'Head','Neck','Body','Front leg','Rear leg','Horse'
]

VAL_DATASET_SIZE = 0.2
TEST_DATASET_SIZE = 0.2

#indexes of points for different segments
HEAD_KEYPOINTS = [
    1, 2, 3, 4, 5, 6, 7, 8, 68, 69, 70, 71
]

NECK_KEYPOINTS = [
    9, 10, 11, 12, 46, 65, 66, 67, 68
]

BODY_KEYPOINTS = [
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 41, 42, 43, 44, 45, 46, 47, 48, 72

]

FRONTLEG_KEYPOINTS = [
    45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
]

REARLEG_KEYPOINTS =[
    21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 
]

HORSE_KEYPOINTS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
]

# learning parameters
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train/test split
TEST_SPLIT = 0.1
# show dataset keypoint plot
SHOW_DATASET_PLOT = True