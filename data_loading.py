from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

import cv2
import random

from PIL import Image

import config
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import sklearn.metrics 
from sklearn.utils import shuffle
import uuid
from PIL import Image
import cv2
import random
import imageio



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# help-function for search patterns like 'IMAGE', 'LM', etc
# and its indexes in tps-file 
def search(pattern, text):
  res = []
  indxs = []
  i = 0
  for line in text:
    if pattern in line:
       res.append(line)
       indxs.append(i)
    i +=1   

  return res, indxs

#help function for replace point->comma and convert to float 
def comma_float(s):
  try:
    return float(s.replace(',','.'))
  except:
    return np.NaN 

def tps_list():

    os.system('./tree_script.sh')

    tps_files = []
    with open('filelist.txt') as fp:
        pathes = fp.readlines()
        for path in pathes:
            dir, filename = os.path.split(path)
            d = {'dir': os.path.join(config.ROOT_DATA_DIRECTORY,dir[2::]), 
                    'file' : filename[:len(filename)-1:]
                }
            tps_files.append(d)          

    for i, file in enumerate(tps_files):
        path = os.path.join(file['dir'],file['file'])
        print(path) 

    os.system('rm filelist.txt')          

    #create empty dataframe
    df=pd.DataFrame(columns=['id','landmarks','imagedir','imagefile'])    

    for file in tps_files:

        dir = file['dir']
        file_name = os.path.join(dir,file['file'])

        with open(file_name, encoding="cp1251") as file:
            lines=file.readlines()
        images, _ = search('IMAGE',lines)
        #ids, _ = search('ID',lines)
        lm, lmixs = search('LM',lines)


        i=0
        for inx in lmixs:
            num = int(lines[inx][3:])
            if (inx+num+1) < len(lines):
                pnts = lines[inx+1:inx+1+num]
                #print(pnts)
            ps=[]
            for p in pnts:
                pnt =  list(map(comma_float, p.split(sep=' ')))
                ps.append(pnt) 

            relpath = images[i][6:-1]
            imagefile = os.path.basename(relpath)
            path = os.path.join(dir,relpath[1:len(relpath)-len(imagefile)-1])

            df = df.append({'id': uuid.uuid4().hex, 
                            'landmarks': ps, 
                            'imagedir':path, 
                            'imagefile': imagefile}, 
                            ignore_index=True)    
            i += 1

    print('total number:\n', df.count())
    print('number of landmarks:\n', len(df['landmarks'][0]))

    return df


#Utility for grayscale detection
def is_gray_scale(img):
    if len(img.shape) < 3: return True, 1
    if img.shape[2]  == 1: return True, 2
    #b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    #if (b==g).all() and (b==r).all(): return True, 3
    return False, 0


# Utility for reading an image and for getting its annotations.
def get_horse(sdf, id):
    path=os.path.join(sdf.iloc[id]['imagedir'],sdf.iloc[id]['imagefile'])
    _, ext = os.path.splitext(path)
    if ext in ['.gif', '.GIF']:
        gif = imageio.mimread(path)            
        # convert form RGB to BGR 
        img_data = cv2.cvtColor(gif[0], cv2.COLOR_RGB2BGR)
    else:
        img_data = cv2.imread(path)
    #If the image is Greyscale convert it to RGB
    gr, _ = is_gray_scale(img_data)
    if gr:       
        img_data=cv2.cvtColor(img_data,cv2.COLOR_GRAY2RGB)
    # If the image is RGBA convert it to RGB.
    if img_data.shape[-1] == 4:
        img_data = img_data.astype(np.uint8)
        img_data = Image.fromarray(img_data)
        img_data = np.array(img_data.convert("RGB"))        
    return img_data

# Utility for getting keypoints
def get_horse_keypoints(sdf, id): 

  return sdf.iloc[id]['landmarks']

def draw_keypoints(keypoints, image):

    h, _, _ = image.shape
    for i in range(0,len(keypoints)):
        image = cv2.circle(image, 
                            center=(int(keypoints[i][0]), int(h-keypoints[i][1])), 
                            radius=h//200, 
                            color=(0, 0, 255), thickness=-1)
        
    return image        

def draw_image_with_keypoints(df, id):
    try:
        image = get_horse(df, id) 
    except Exception as e:
        print(str(e),  id, os.path.join(df.iloc[id]['imagedir'],df.iloc[id]['imagefile']))       
    else:
        image = draw_keypoints(list(df['landmarks'][id]), image)
        save_name = f"{df.iloc[id]['imagefile'].split('.')[0]}_out"
        cv2.imwrite(f"outputs/{save_name}.jpg", image)


if __name__ == "__main__":
    tps_df=tps_list()    
    #print(tps_df.head(31))
    #print(tps_df.iloc(0)[0])
    
    random_idxs = []
    random.seed(42)
    for i in range(5):
        n = random.randint(1,len(tps_df))
        random_idxs.append(n)
    print(random_idxs)
    #for idx in random_idxs:
    for idx in range(len(tps_df)):
        if (idx % 20) == 0:
            print('i=',idx)
        draw_image_with_keypoints(tps_df, idx)
    #draw_image_with_keypoints(tps_df, 30)