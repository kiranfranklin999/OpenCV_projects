# importing libraries
import cv2
import time
import imutils

cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame=None
area = 500

while True:
    _,img = cam.read()  # reading the info from the camera
    text = "Normal"
    img = imutils.resize(img, width=500) # maintains the aspect ratio of width 500 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting the RGB image to gray scale image
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0) # applying Gaussian Smoothing on the input source image
    
    if firstFrame is None: ## capturing the frist frame 
            firstFrame = gaussianImg
            continue
    
    ##### peforming DST : Frame Differencing and Summing Technique 
    imgDiff = cv2.absdiff(firstFrame, gaussianImg) # frame difference btw frist frame and latest
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] # thrsholding the difference frame
    threshImg = cv2.dilate(threshImg, None, iterations=2) # applies a morphological filter

    ##### find and drawing contours of the objects in the image
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
            if cv2.contourArea(c) < area:  ## if Contour area image is less than orginal area normal else moved
                    continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Moving Object detected"
#     print(text)
    cv2.putText(img, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
