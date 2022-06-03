**Face Recognition**

General steps of face recognition:

Step 1: Face Detection: Look at the picture and find a face in it.
Step 2: Data Gathering: Extract unique characteristics of face that it can use to differentiate one from another person, like eyes, mouth, nose, etc.
Step 3: Data Comparison: Despite variations in light or expression, it will compare those unique features to all the features of all the people you know.
Step 4: Face Recognition

*How to Perform this using computer/ OpenCV?*

OpenCV has three built-in face recognizers and thanks to its clean coding, you can use any of them just by changing a single line of code. Here are the names of those face recognizers and their OpenCV calls:

EigenFaces – cv2.face.createEigenFaceRecognizer()
FisherFaces – cv2.face.createFisherFaceRecognizer()
Local Binary Patterns Histograms (LBPH) – cv2.face.createLBPHFaceRecognizer()

In this project we'll be using Last method because it's the improvised version of other two over their cons.


*How to Build your face recogniers?*

step 1: Create the data for training.

*python create_data.py --haar_file <file path for haarcascade_frontalface_default.xml> --dataset   <Main folder path where images data to be stored> --sub_data <sub folder name where specific face data to be stored>*

Run this code N times of different person's face data u wanted to generate to train the model

step 2: Training.

*python train.py --dataset <Main folder path where images data to be stored>*

It saves model named "classifier.yml" in the current working directory

step 3: Recognise or predict

*python face_recognize.py --haar_file <file path for haarcascade_frontalface_default.xml> --dataset <Main folder path where images data to be stored> --model <file path of previosly developed model>*

