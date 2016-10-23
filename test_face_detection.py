## Created by: Abdulaziz Alaa
##
## simple script to calculate the distance between the detected face rect using an algorithm and the actual face rect
## Calculate the Error

import numpy as np
import cv2
import pandas as pd
import math

opencv_path = "/home/abdulaziz/workspace/OpenCV/opencv/"
dataset_path = "../../data-sets/"

face_cascade = cv2.CascadeClassifier(opencv_path+'data/haarcascades/haarcascade_frontalface_default.xml')

imgs = pd.read_csv(filepath_or_buffer=dataset_path+"muct-master/muct-landmarks/muct76-opencv.csv", sep=",", delimiter=None)
img_names = imgs.iloc[:,0]
img_landmarks = imgs.iloc[:, 2:]

face_img = cv2.imread(dataset_path+"muct-master/jpg/"+img_names[0]+".jpg")

num_rows = img_landmarks.shape[1]
i = 0
points = []
while i<num_rows:
    point = (int(img_landmarks.iloc[0, i]), int(img_landmarks.iloc[0, i+1]))
    points.append(point)

    cv2.circle(face_img, point, 1, (0,0,255), -1)

    i=i+2

points = np.array(points)
rx, ry, rw, rh = cv2.boundingRect(points) #real points

cv2.rectangle(face_img,(rx, ry),(rx+rw, ry+rh),(0,255,0),3)


#### viola jones detector
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for(px, py, pw, ph) in faces: #pridected point
    cv2.rectangle(face_img,(px,py),(px+pw, py+ph),(255,0,0),2)


### calculating distance
rP1 = (rx, ry) #real first point
pP1 = (px, py) #pridected first point

rP2 = (rx+rw, ry) #real first point
pP2 = (px+pw, py) #pridected first point

rP3 = (rx, ry+rh) #real first point
pP3 = (px, py+ph) #pridected first point

rP4 = (rx+rw, ry+rh) #real first point
pP4 = (px+pw, py+ph) #pridected first point

distance = math.sqrt( math.pow(rP1[0]-pP1[0], 2) + math.pow(rP2[0]-pP2[0], 2) + math.pow(rP3[0]-pP3[0], 2) + math.pow(rP4[0]-pP4[0], 2) +
                     math.pow(rP1[1]-pP1[1], 2) + math.pow(rP2[1]-pP2[1], 2) + math.pow(rP3[1]-pP3[1], 2) + math.pow(rP4[1]-pP4[1], 2))/2

print "Error between Real and Predicted is " + str(distance)

cv2.imshow('face rects', face_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
