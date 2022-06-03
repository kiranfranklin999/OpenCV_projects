import cv2, numpy, os
from argparse import ArgumentParser




def train(datasets):
# datasets = 'dataset'
    print('Training...')
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
                #print(labels)
            id += 1
    (width, height) = (130, 100)

    (images, labels) = [numpy.array(lis) for lis in [images, labels]]

    #print(images, labels)
    model = cv2.face.LBPHFaceRecognizer_create()
    #model =  cv2.face.FisherFaceRecognizer_create()
    model.train(images, labels)

    model.save("classifier.yml")



if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--haar_file", type=str, default="F:\\kiran\\AI_internship\\GitHub_projects\\OpenCV_projects\\Face_recognition\\haarcascade_frontalface_default.xml", help="haarcascade_frontalface_deafult.xml file path")
    parser.add_argument("--dataset", type=str, default=os.path.join(os.getcwd(),"dataset"), help="Main folder path where images data to be stored")
    args = parser.parse_args()

    train(args.dataset)