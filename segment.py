# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oomDOYOE6yeQyMb0NS1KC7AdJxY4OxwG
"""

# Import Required Libraries
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Check if a rectangle present inside other
# helps is selecting the outermost rectangle
def rectContains(rect, pt):
    ret = rect[0] <= pt[0] <= rect[0] + rect[2] and rect[1] <= pt[1] <= rect[1] + rect[3]
    return ret

# Find all the biggest rectangles/contours in the image
def refine_contours(list_coord):
    ret_cnt = list_coord.copy()
    
    for i in list_coord:
        for j in list_coord:
            if rectContains(i, [j[0], j[1]]) and rectContains(i, [j[0] + j[2], j[1]]) and rectContains(i, [j[0], j[1] + j[3]]) and rectContains(i, [j[0] + j[2], j[1] + j[3]]) and i != j:
                if j in ret_cnt:
                    ret_cnt.remove(j)
                    
    return ret_cnt

# Remove shadow from the imaage
def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
        
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov


# Find list of Contours
def Find_list_contours(contours, diff, image_result):
    list_contours = []
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area < 0.9 * (image_result.shape[0] * image_result.shape[1])):
            max_area = max(max_area,area)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if(area < 0.9 * (image_result.shape[0] * image_result.shape[1])):
            x, y, w, h = cv2.boundingRect(cnt)
            if w > image_result.shape[1] / diff and h > image_result.shape[0] / diff and area>=max_area/20:
                cv2.rectangle(image_result,(x,y),(x+w,y+h),(0,255,0),2)
                list_contours.append([x, y, w, h])
    cv2_imshow("image_result***",image_result)
    return list_contours

# Find all contours of words present in image if any present
# This is special factor of our model that it can also read if multiple words given that they are written in same line. 
def find_contours_word(img):

    # img = cv2.imread(path)

    img = shadow_remove(img)# removes shadow in image if any
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(img, (3,3), sigmaX = 0)
    
    ret, image_result = cv2.threshold(gaussian_blur, 220, 255, cv2.THRESH_BINARY)
    image_result = cv2.copyMakeBorder(image_result, 10, 10, 10, 10, cv2.BORDER_CONSTANT,None,255)
    # cv2_imshow("image_result",image_result)
    kernel = np.ones((3, 3), np.uint8)
    image_result = cv2.dilate(image_result, kernel, iterations = 1)
    image_result = cv2.erode(image_result, kernel, iterations = 1)

    # to fill the gap between letters of word if present
    image_result = cv2.erode(image_result, kernel, iterations = 5)
    contours, hierarchy = cv2.findContours(image_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contours of multiple word and ignore if small area contours if detected
    list_contours = Find_list_contours(contours, 30, image_result)
    
    # helps to rectangle which are bounded by a bigger rectangles
    final_contours = refine_contours(list_contours)

    final_contours = sorted(final_contours, key=lambda ctr: ctr[0])
    # cv2_imshow("img##",img)
    return final_contours

# To break characters and returns a array of letters
def segmentation(img):    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(img, (3, 3), sigmaX = 0)
    
    kernel = np.ones((3, 3), np.uint8)
    image_result = cv2.dilate(gaussian_blur, kernel, iterations = 1)
    image_result = cv2.erode(gaussian_blur, kernel, iterations = 1)

    ret, image_result = cv2.threshold(gaussian_blur, 230, 255, cv2.THRESH_BINARY)
    
    image_result = cv2.copyMakeBorder(image_result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, 255)
    image_median = cv2.medianBlur(image_result, 3)
    # function of erosion and dilation is reversed in this image as it is written in balck and background is white.
    dilation1 = cv2.dilate(image_median, kernel, iterations = 1)
    erosion1 = cv2.erode(dilation1, kernel, iterations = 1)
    
    pre = erosion1.copy()
    # cv2_imshow("pre",pre)
    # kernel of size (1, 9) helps in obtaining horizontal lines in image
    #  Then we picked the largest line amoung them (which repersent the upper line present in devnagiri
    # language) and removed it from orignal image so that segmentation can be performed
    kernel = np.ones((1, 9), np.uint8)
    dilation = cv2.dilate(erosion1, kernel, iterations = 2)
    erosion = cv2.erode(dilation, kernel, iterations = 2)
    
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(erosion, kernel, iterations = 1)
    
    # This code block computes the maximum area horizontal line
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    target_contour = None 
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.9 * (img.shape[0] * img.shape[1]) and area > max_area:
            target_contour = contour
            max_area = area



    w = erosion.shape[0]
    h = erosion.shape[1]
    #  replaces the top line only if the word length is not one letter
    if(w>2*h or h>2*w):
        try:
            cv2.fillPoly(erosion1, pts = [target_contour], color = (255, 255, 255))
        except:
            pass
    # cv2_imshow("erosion1",erosion1)
    kernel = np.ones((3, 5), np.uint8)
    erosion1 = cv2.erode(erosion1, kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(erosion1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    pre = 255 - pre
    # Inverting the whole image for prediction
    
    letters = []# will take all the individual letter images
    
    list_cnt = Find_list_contours(contours, 200, image_result)
    # helps to find all contours
    
    final_cnt = refine_contours(list_cnt)
    final_cnt = sorted(final_cnt, key=lambda ctr: ctr[0])
    
    for coord in final_cnt:
        [x, y, w, h] = coord
        # print("x, y, w, h = ",x, y, w, h)
        # print(image_result.shape[0],image_result.shape[1])
        temp = pre[(int)(max(0, y - 0.2 * h)) : (int)(min(pre.shape[0], y + 1.2 * h)), (int)(max(0, x - 0.05 * w)) : (int)(min(pre.shape[1], x + 1.05 * w))]
        # cv2_imshow("temp",temp)
        # print((int)((image_result.shape[1]/image_result.shape[0])*4))
        if temp.shape[0] * temp.shape[1] < 1000 or w <= image_result.shape[1] / (int)((image_result.shape[1]/image_result.shape[0])*4) or h <= image_result.shape[0] / 5:
            continue
        # giving some black border to the images
        temp = cv2.copyMakeBorder(temp, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
        letters.append(temp)
        
    return letters

def cv2_imshow(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
