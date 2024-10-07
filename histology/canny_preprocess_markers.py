import cv2
import numpy as np 
from skimage import feature
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

def preprocess_edges(image, thresholds, cell_areas):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3,3), 0)

    # Detect edges
    edges = cv2.Canny(image, thresholds[0], thresholds[1], L2gradient=True)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Gaussian image")
    ax[1].imshow(edges, cmap=plt.cm.gray)
    ax[1].set_title("Edged image")

    # Detect contours 
    # Hierarchy is organised as: [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    print(len(contours), "objects were found in this image.")

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1)
    ax[2].imshow(contour_image, cmap=plt.cm.gray)
    ax[2].set_title("Contour image")

    # Find parent contours - outer membrane - if there are children and area matches cell area
    for c, contour in enumerate(contours):
        if hierarchy[c][2] != -1 and np.abs(cv2.contourArea(contour) - cell_areas[0]) < 200:             

            # Find child with largest area
            child_idx = hierarchy[c][2]
            max_area = 0
            max_child = None
            while child_idx != -1:
                child_hier = child_idx  # next child in hierarchy
                area = cv2.contourArea(contours[child_idx]) # area of child
                if area > max_area:
                    max_area = area
                    max_child = child_idx
                max_area = max(area, max_area)
                child_idx = child_hier[0] # TODO: check this is correct

            # Fill the contour with the largest area 
            if max_child is not None:
                    cv2.drawContours(contour_image, [contours[child_idx]], -1, [0,0,255], -1) # draw inner membrane

    ax[3].imshow(contour_image, cmap=plt.cm.gray)
    ax[3].set_title("Filled membranes")
                         
           



    


    
    



def get_largest_cnt(img, point):
    """
    This code returns the largest contour in the given image which encapsulates the given "point"
    """
    height, width = img.shape[0:2]
    
    
    if point==(0,0):
        x=width/2
        y=height/2
        point=(x,y)
    distMin=np.infty
    contours=cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    #Finding the centered contour in the image
    cnt_center = sorted(contours, key=cv2.contourArea)[-1]
    for i in range(0,len(contours)):
        cnt = sorted(contours, key=cv2.contourArea)[-1-i]
        
        #Check if point is in contour
        dist=cv2.pointPolygonTest(cnt, point, False)
        if dist>0:
            print("cnt:"+str(i))
            cnt_center=cnt
#             distMin=dist
            break
        else:
            cnt_center=cnt
        
#     print(distMin)
    return cnt_center


def mask_largest_cnt(img, otsu_mask, point):
    """
    Mask the largest contour
    """
    cnt=get_largest_cnt(otsu_mask, point)
    mask = np.zeros((np.shape(img)[0],np.shape(img)[1]), np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
#     if prev_cnt.size>4:
#         prev_mask_scaled=np.zeros((np.shape(img)[0],np.shape(img)[1]), np.uint8)
#         prev_cnt=scale_contour(prev_cnt,scale)
#         cv2.drawContours(prev_mask_scaled, [prev_cnt], -1, 255, -1)
# #         mask=(gray2binary(mask)*gray2binary(prev_mask_scaled)).astype('uint8')
#         mask=np.logical_and(mask, prev_mask_scaled).astype('uint8')
    masked=(gray2binary(mask)*img).astype('uint8')

#     dst = cv2.bitwise_and(image, image, mask=mask)
    return masked


def scale_contour(cnt, scale):
    """
    Scale the contour
    """
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled