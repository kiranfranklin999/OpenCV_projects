# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:54:22 2020

@author: newdi
"""

import cv2, imutils, time

haar_cascade = cv2.CascadeClassifier("Mouth.xml")
haar_cascade1 =cv2.CascadeClassifier("Nariz.xml")
t1="mask detected"
t2="no mask detected"
t3=" Improper mask detected. Wear mask properly, Both Nose and Mouth should be covered"
cam =cv2.VideoCapture(0)
time.sleep(1)
while True:
    _,img = cam.read() # reading the info from the camera
    # img = imutils.resize(img, width=500)
    bilateral_filtered_image = cv2.bilateralFilter(img, 7, 150, 150)  ## removing the noise 
    grayImg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converting the RGB image to gray scale image
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0) # applying Gaussian Smoothing on the input source image
    gaussianImg = cv2.dilate(gaussianImg, None, iterations=2) # applies a morphological filter
    nose = haar_cascade1.detectMultiScale(gaussianImg,1.3,4) # detecting the nose using the pretrained classifer and returns the co-ordinates 
    mouth = haar_cascade.detectMultiScale(gaussianImg,1.3,4) # detecting the mouth using the pretrained classifer and returns the co-ordinates
    print(nose,mouth)
    ## notifying the programmer the info based on the previous detected info
    if ( len(mouth)!=0) and len(nose)!=0:
        cv2.putText(img, t2, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(mouth)!=0 and len(nose)==0:
        cv2.putText(img, t3, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(img, t1, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    ## marking the detected features of the face
    for (x,y,w,h) in mouth:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    for (x,y,w,h) in nose:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("face" , img)
    key = cv2.waitKey(10)
    
    # terminating the code
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
    