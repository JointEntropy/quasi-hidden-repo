import os
import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

def erode(mask, kernel_size=2, iters=1):
    return cv2.erode(mask, np.ones((kernel_size, kernel_size),np.uint8), iterations=iters)

def polish(mask, kernel_size=3, iters=1):
    return cv2.morphologyEx(mask,cv2.MORPH_OPEN,
                    kernel=np.ones((kernel_size,kernel_size)), 
                    iterations=iters) 

def distance_watershed(mask, distance=None, kernel_size=3, debug=False):
    """
    Input - one channel boolean mask with binary segmentation.
    Watershed with distances.
    """
    assert len(mask.shape)==2
    assert mask.dtype==np.bool
    
    if not distance:
        distance = ndi.distance_transform_edt(mask)
    if debug:
        plt.imshow(distance)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((kernel_size, kernel_size)),
                                labels=mask)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=mask)
    return labels

def get_boundary(img, boundary=0.001, dist_transform=5):
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # prepare step
    thresh = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) 
    
    # step 1
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,dist_transform)
    ret, sure_fg = cv2.threshold(dist_transform, boundary*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # step 2
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # step 3
    markers = markers+1
    markers[unknown==255] = 0
    border = np.zeros_like(img)
    markers = cv2.watershed(img,markers)
    border[markers == -1] = [255,255,255]

    cv2.rectangle(border,(0,0),(border.shape[1],border.shape[0]),(0,0,0),2)
    return mask


def transform_masks(transform, 
                    dpath='data/stage1_train', 
                    aggregate=True, 
                    transform_name='eroded',
                    debug=False):
    '''
    For each sample in dpath apply transform for each  mask of that sample and save in 
    contigious dir called by transform name.
    Optionally aggregate all results in one image.
    In debug mode do transform on one sample and return path.
    '''
    for sample_dir in tqdm(os.listdir(dpath)):
        sample_path = os.path.join(dpath, sample_dir)
        
        os.makedirs(os.path.join(sample_path, transform_name), exist_ok=True)
        if aggregate:
            aggregated = None
            for mask_id in os.listdir(os.path.join(sample_path, 'masks')):
                mask = cv2.imread(os.path.join(sample_path,'masks',mask_id))
                transformed = transform(mask)
                aggregated =  aggregated+transformed if aggregated is not None else transformed
                cv2.imwrite(os.path.join(sample_path, transform_name, mask_id), transformed)
            aggregated = np.minimum(aggregated,255)
            cv2.imwrite(os.path.join(sample_path, transform_name+'.png'), aggregated)
            #plt.imshow(aggregated)
        else:
            for mask_id in os.listdir(os.path.join(sample_path, 'masks')):
                mask = cv2.imread(os.path.join(sample_path,'masks',mask_id))
                transformed = transform(mask)
                cv2.imwrite(os.path.join(sample_path, transform_name, mask_id), transformed)
        if debug:
            return os.path.join(dpath, sample_dir)        

if __name__ == '__main__':
    dir_ = transform_masks(transform=(lambda x: distance_transform_edt(x[:,:,0])), 
                       transform_name='distance',
                       debug=True)
	plt.subplots(1,2, figsize=(15,5))
	# On aggregated mask
	k = plt.imread(os.path.join(dir_,'dummy.png'))[:,:,0]
	k = distance_transform_edt(k)
	plt.subplot(1,2,1)
	plt.imshow(k)

	# One single masks
	plt.subplot(1,2,2)
	plt.imshow(plt.imread(dir_+'/distance.png'))



