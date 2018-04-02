import os
import numpy as np
import cv2
from tqdm import tqdm

def get_boundary(img, boundary=0.001, dist_transform=5):
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)  # удаляем шумы внутри регионов(для переднего удаляем чёрные отверстия, для заднего плана - светлые)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,dist_transform)
    
    ret, sure_fg = cv2.threshold(dist_transform, boundary*dist_transform.max(),255,0)
    # Finding unknown region                       # получаем границу - как разницу между задним планом и передним планом.
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    mask = np.zeros_like(img)
    #mask[markrs == -1] = [255,0,0]
    markers = cv2.watershed(img,markers)
    mask[markers == -1] = [255,255,255]

    cv2.rectangle(mask,(0,0),(mask.shape[1], mask.shape[0]),(0,0,0),2)
    return mask

def prepare_borders(dpath='data/stage1_train', **get_boundary_args):
    for i, sample_dir in enumerate(os.listdir(dpath)):
        for sample in tqdm(os.path.join(dpath, sample_dir)):
            os.makedirs(os.path.join(dpath, sample, 'borders'), exist_ok=True)
            border = None
            for mask_id in os.listdir(os.path.join(dpath, sample, 'masks')):
                img = cv2.imread(os.path.join(dpath,sample,'masks',mask_id))
                border_tmp =  get_boundary(img, **get_boundary_args)
                border =  border_tmp+ border if border is not None else border_tmp 
                #border = cv2.erode(img,None,iterations = 2)
                cv2.imwrite(os.path.join(dpath, sample, 'borders', mask_id), border_tmp)
            cv2.imwrite(os.path.join(dpath, sample, 'border.png'), border)
