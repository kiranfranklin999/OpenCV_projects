#importing libraies
import numpy as np
import imutils
import cv2, os
import time
from argparse import ArgumentParser



def main(prototxt,model):
    #storing pretrained models
    # prototxt = "MobileNetSSD_deploy.prototxt.txt"
    # model = "MobileNetSSD_deploy.caffemodel"
    confThresh = 0.2
    #init of classes
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
    #selection of colors rand based on leng of class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    print("Loading model...")
    #loading the model
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("Model Loaded")
    print("Starting Camera Feed...")
    #acessing the camera
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        _,frame = vs.read()
        #resizing the image
        frame = imutils.resize(frame, width=500)
        #getting ht and  width for plotting the frame
        (h, w) = frame.shape[:2]
        imResizeBlob = cv2.resize(frame, (300, 300))
        #rgb to blob
        blob = cv2.dnn.blobFromImage(imResizeBlob,0.007843, (300, 300), 127.5)
        #passing the blob image to pre traned model
        net.setInput(blob)
        detections = net.forward()# image fo classifiction
        detShape = detections.shape[2]#getting the shape
        for i in np.arange(0,detShape):
            confidence = detections[0, 0, i, 2]# identify thr confidence level
            if confidence > confThresh:     
                idx = int(detections[0, 0, i, 1])# identify the class
                print("ClassID:",detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])# drawing the frame arounf the image
                (startX, startY, endX, endY) = box.astype("int")
                
                label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                if CLASSES[idx]=="bottle":
                    label = "{}: {:.2f}% i need water".format(CLASSES[idx],confidence * 100)
                if startY - 15 > 15:
                    y = startY - 15
                else:
                    startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    vs.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--prototxt", type=str, default=os.path.join(os.getcwd(),"MobileNetSSD_deploy.prototxt.txt"), help="file path to Caffe ‘deploy’ prototxt file")
    parser.add_argument("--model", type=str, default=os.path.join(os.getcwd(),"MobileNetSSD_deploy.caffemodel"), help="pretrained model path")
    args = parser.parse_args()
    print(args)
    main(args.prototxt,args.model)

