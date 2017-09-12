import sys
import os
import dlib
import glob
import numpy  
import cv2
from skimage import io
from scipy.spatial import distance
from imutils import face_utils
import os, sys
from PIL import Image
import face_recognition

image1 = sys.argv[1] + '.jpg'
image2 = sys.argv[2] + '.jpg'

def imageResize(item):
    basewidth = 800
    img = Image.open(item)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(item)

imageResize(image1)
imageResize(image2)

predictor_path = "C:\\Users\\yfu027\\PycharmProjects\\OpenCV\\FaceDetect\\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "C:\\Users\\yfu027\\PycharmProjects\\OpenCV\FaceDetect\\face_rec_model_path\\dlib_face_recognition_resnet_model_v1.dat"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

resultArray = []

# Now process all the images
img = cv2.imread(image1)
faces = detector(img,1)
if (len(faces) > 0):
    for k,d in enumerate(faces):
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
        shape = landmark_predictor(img,d)

        face_descriptor = facerec.compute_face_descriptor(img, shape)		
        resultArray.append(numpy.array(face_descriptor))

        for i in range(68):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y),5,(0,255,0), -1, 8)
            cv2.putText(img,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
cv2.imshow('Frame',img)

img = cv2.imread(image2)
faces = detector(img,1)
if (len(faces) > 0):
    for k,d in enumerate(faces):
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()+100),(255,255,255))
        shape = landmark_predictor(img,d)

        face_descriptor = facerec.compute_face_descriptor(img, shape)		
        resultArray.append(numpy.array(face_descriptor))

        for i in range(68):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y),5,(0,255,0), -1, 8)
            cv2.putText(img,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,2555,255))
        
        dst = distance.euclidean(resultArray[0],resultArray[1])
        result = "Match"
        if dst > 0.45: 
            result = 'Not Match'
          
        cv2.putText(img,str('Predict Result:' + result),(d.left(), d.top()),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0))
        # imageone = face_recognition.load_image_file("test1.jpg")
        # imageone_face_encoding = face_recognition.face_encodings(imageone)[0]
        # imagetwo = face_recognition.load_image_file("test2.jpg")
        # imagetwo_face_encoding = face_recognition.face_encodings(imagetwo)[0]
        # face_names = []      
        # # See if the face is a match for the known face(s)
        # match = face_recognition.compare_faces([imageone_face_encoding], imagetwo_face_encoding)
        # if match[0]:
        #     print('true')

cv2.imshow('Frame2',img)

dst = distance.euclidean(resultArray[0],resultArray[1])
print(dst)

cv2.waitKey(0)
