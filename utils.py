import config
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from config import RECTANGLES_ATTRIBUTES as horse_segments_names
from PIL import Image, ImageDraw, ImageFont

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def find_indexes(lst, x):
    indexes = []
    for i in range(len(lst)):
        if lst[i] == x:
            indexes.append(i)
    return indexes

def max_value_with_index(a, b):
    max_value = float('-inf')
    max_index = None

    for index in b:
        if index < len(a):
            if a[index] > max_value:
                max_value = a[index]
                max_index = index

    return max_value, max_index


def predict_horse_segments(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        outputs = model(image) # get the predictions on the image
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > detection_threshold]
    # get all the predicted bounding boxes
    bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)
    # get all the predicited class names
    labels = outputs[0]['labels'].cpu().numpy()
    pred_classes = [horse_segments_names[labels[i]] for i in thresholded_preds_inidices]
    scores_final = [scores[i] for i in thresholded_preds_inidices]  
    return boxes, pred_classes, scores_final

def get_max_score_segments(boxes, classes, scores):
    b =[]
    c =[]
    s =[]

    for i in horse_segments_names:
        idxs = find_indexes(classes, i)
        if len(idxs) > 0:
            _, idx = max_value_with_index(scores, idxs)
            b.append(boxes[idx]) 
            c.append(classes[idx])
            s.append(scores[idx])
    return b, c, s    

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        outputs = model(image) # get the predictions on the image
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > detection_threshold]
    # get all the predicted bounding boxes
    bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)    
    # get all the predicited class names
    labels = outputs[0]['labels'].cpu().numpy()
    pred_classes = [coco_names[labels[i]] for i in thresholded_preds_inidices]
    return boxes, pred_classes

def draw_keypoints(keypoints, image):

    h, _, _ = image.shape
    for i in range(0,len(keypoints)):
        image = cv2.circle(image, 
                            center=(int(keypoints[i][0]), int(h-keypoints[i][1])), 
                            radius=h//100, 
                            color=(0, 0, 255), thickness=-1)
        
    return image     


def draw_horse_boxes(boxes, classes, image):
    for i, box in enumerate(boxes):
        color = (255,0,0) #COLORS[coco_names.index(classes[i])]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, 
                    lineType=cv2.LINE_AA)
        #pil_image = Image.fromarray(image)
        #draw = ImageDraw.Draw(pil_image)
        #draw.text((int(box[0]), int(box[1]-5)), 'лошадь')
        # Convert back to Numpy array and switch back from RGB to BGR
        #image = np.asarray(pil_image)        


    return image    

def copy_horse_boxes(boxes, classes, scores, image, threshold=0.6):
    images=[]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])  
        """
        if scores[i] < threshold:
            w = x2 - x1
            x1 = max(0, x1 - int(w*0.1))
            x2 = min(image.shape[1], x2 + int(w*0.1))
        """    
        im = image[y1:y2,x1:x2].copy()
        #print(im)
        images.append(im)
    return images      

def get_segment_keypoints(keypoints, segment_keypoints_indices):
    kps=[]
    for idx in segment_keypoints_indices:
        kps.append(keypoints[idx-1])
    return kps     

def embed_keypoints_to_box(keypoints, box, height):
    kps=[]
    h = box[3]-box[1]    
    for kp in keypoints:
        x = int(kp[0])
        y = int(kp[1])
        yr = height - y
        v = 1
        if (x > box[2]) or (yr > box[3]):
            v = 0
        else:
            if (x < box[0]) or (yr < box[1]):
                v = 0
        x = x - box[0]
        yr = yr - box[1]
        y = h - yr
        kps.append([x,y,v])                                
    return kps        

def get_visible_keypoints(keypoints):
    kps=[]
    for kp in keypoints:
        if kp[2]:
            kps.append([kp[0], kp[1]])
    return kps          

def get_all_segments_keypoints(all_kps, boxes, classes, height, idx, draw=False):
    all_ekps={}
    for b, c in zip(boxes, classes):
        cl = True
        match c:
            case 'Head':
                ski = config.HEAD_KEYPOINTS
            case 'Neck':
                ski = config.NECK_KEYPOINTS           
            case 'Body':
                ski = config.BODY_KEYPOINTS          
            case 'Front leg':
                ski = config.FRONTLEG_KEYPOINTS            
            case 'Rear leg':
                ski = config.REARLEG_KEYPOINTS   
            case 'Horse':
                ski = config.HORSE_KEYPOINTS                          
            case _:
                cl = False         
        if cl:
            kps=get_segment_keypoints(all_kps, ski)
            ekps = embed_keypoints_to_box(kps, b, height)            
            all_ekps[c] = ekps 


            if draw:
                vkps = get_visible_keypoints(ekps)
                file_path=f"outputs/{c.lower()}/{idx}.jpg"
                image = cv2.imread(file_path)
                image = draw_keypoints(vkps, image)
                file_path=f"outputs/{c.lower()}/{idx}_kps.jpg"
                cv2.imwrite(f"{file_path}", image)              

    return all_ekps       

def valid_keypoints_plot(image, outputs, orig_keypoints, epoch):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    # just get a single datapoint from each batch
    img = image[0]
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    h, _, _ = img.shape
    output_keypoint = output_keypoint.reshape(-1, 3)
    orig_keypoint = orig_keypoint.reshape(-1, 3)
    for p in range(output_keypoint.shape[0]):
        plt.plot(output_keypoint[p, 0], h-output_keypoint[p, 1], 'r.')
        plt.plot(orig_keypoint[p, 0], h-orig_keypoint[p, 1], 'b.')
    plt.savefig(f"{config.ROOT_OUTPUT_DIRECTORY}/val_epoch_{epoch}.png")
    plt.close()     