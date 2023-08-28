import argparse
import math

import config
import cv2
import numpy as np
import torch
import torchvision
import utils
from PIL import Image
from torchvision.models.detection import (RetinaNet_ResNet50_FPN_V2_Weights,
                                          retinanet_resnet50_fpn_v2)
from torchvision.models.detection.retinanet import (
    RetinaNetClassificationHead, RetinaNetHead)
from utils import get_max_score_segments


def prepare_segments(file_path, out_name, threshold=0.5):
    model = retinanet_resnet50_fpn_v2(weights=None, box_score_thresh=0.7)    
    num_classes = config.SEGMENTS_NUMBER    
    # replace classification layer
    # replace classification layer
    out_channels = model.head.classification_head.conv[0].out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits    
    model.load_state_dict(torch.load(f'{config.ROOT_SEGMENTATION_MODEL_DIRECTORY}/{config.SEGMENTATION_MODEL_PATH}'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the model onto the computation device
    model.eval().to(device)

    image = Image.open(file_path).convert('RGB')
    # a NumPy copy for OpenCV functions
    image_array = np.array(image)
    # convert to OpenCV BGR color format
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    h,w,_ = image_array.shape

    img = image.copy()
    img=img.resize((256,256))
    # get the bounding boxes and class labels
    boxes, classes, scores = utils.predict_horse_segments(img, model, device, threshold)
    #print(boxes, classes, scores)
    b, c, s = get_max_score_segments(boxes, classes, scores)
    b = utils.resize_boxes(b, from_size=(256,256), to_size=(w,h))


    # get the final image
    results = utils.copy_horse_boxes(b, c, s, image_array)
    for i in range(len(results)):
        img = results[i]
        save_name = f"{c[i].lower()}/{out_name}"
        cv2.imwrite(f"outputs/{save_name}.jpg", img)

    return b, c, h        

