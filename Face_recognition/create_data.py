import cv2, os ,time
from argparse import ArgumentParser
# haar_file = 'haarcascade_frontalface_default.xml'
# datasets = 'dataset'  
# sub_data = 'Franklin'     


def main(haar_file,datasets,sub_data):
    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    (width, height) = (130, 100)   


    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0) 
    time.sleep(1)
    count = 1
    while count < 31:
        print(count)
        _, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # print("gray:",gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('%s/%s.png' % (path,count), face_resize)
        count += 1
        
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--haar_file", type=str, default="F:\\kiran\\AI_internship\\GitHub_projects\\OpenCV_projects\\Face_recognition\\haarcascade_frontalface_default.xml", help="haarcascade_frontalface_deafult.xml file path")
    parser.add_argument("--dataset", type=str, default=os.path.join(os.getcwd(),"dataset"), help="Main folder path where images data to be stored")
    parser.add_argument("--sub_data", type=str, default="franklin", help="sub folder name where specific face to b stored")
    args = parser.parse_args()

    main(args.haar_file,args.dataset,args.sub_data)