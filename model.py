import config
import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                          keypointrcnn_resnet50_fpn)
from torchvision.models.detection.rpn import AnchorGenerator


class HorseKeypointResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad, keypoints_number):
        super(HorseKeypointResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
            #anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0,3.0, 4.0))
                                                                                             
            #self.model = keypointrcnn_resnet50_fpn(pretrained=False,
            #                                        pretrained_backbone=True,
            #                                        num_keypoints=keypoints_number,
            #                                    rpn_anchor_generator=anchor_generator)
            #self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, num_keypoints=keypoints_number)
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
            #self.model = keypointrcnn_resnet50_fpn(weights=None, num_keypoints=keypoints_number)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layer
        self.l0 = nn.Linear(2048, keypoints_number*3)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0