import numpy as np
import cv2
import argparse
import imutils
import math

def four_point(img):
    r = 500/ img.shape[1]
    dim= (500, int(img.shape[0]*r))
    orig = img
    img = cv2.resize(img,dim, interpolation= cv2.INTER_AREA)
    cv2.imshow('resized image', img)
    cv2.waitKey(0)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('canny image', img)
    cv2.waitKey(0);
    img = cv2.GaussianBlur(img, (5,5), 0)
    cv2.imshow('canny image', img)
    cv2.waitKey(0);
    img = cv2.Canny(img, 100,200)

    cv2.imshow('canny image', img)
    cv2.waitKey(0);

    image,contours,heirarchy =  cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        something = cv2.approxPolyDP(contour, 0.02*perimeter, True)

        if len(something)==4:
            contours= something
            break
    cv2.drawContours(img,[contours],-1,(150,200,0),5)
    cv2.imshow('Drawn contour',img)
    cv2.waitKey(0)

    contours = contours.reshape(4,2)/r;
    contours = contours.astype(np.float32)
    (tr,tl,bl,br) = contours

    widthA= math.sqrt(((tl[0]-tr[0])**2)+((tl[1]-tr[1])**2))
    widthB= math.sqrt(((bl[0]-br[0])**2)+((bl[1]-bl[1])**2))

    width = max(int(widthA),int(widthB))

    heightA= math.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    heightB = math.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))

    height = max(int(heightA),int(heightB))
    
    final = np.array([[width-1,0],[0,0],
                     [0,height-1],[width-1,height-1]],np.float32)

    map = cv2.getPerspectiveTransform(contours, final)
    warp = cv2.warpPerspective(orig,map,(width,height))

    return warp


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",help="Path to the image")
args = vars(ap.parse_args())
if(len(args)==0):
    print 'No arguments passed'
    exit

image = cv2.imread(args["image"])

warped = four_point(image);

cv2.imshow('Input', image)
warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
warped = cv2.GaussianBlur(warped, (5,5), 0)
warped = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('Output',warped)

cv2.waitKey(0)

