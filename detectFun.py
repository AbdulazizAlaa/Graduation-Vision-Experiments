import pandas as pd
import cv2
import sys
import math
sys.path.append("/home/yomna/GraduationProject/openface/openface")
from align_dlib import AlignDlib
import numpy as np
from numpy import genfromtxt


def muchtDatasetRects(my_data):
    muchtRects = []

    for row in my_data[1:]:
        points = [(int(float(row[i+2])), int(float(row[i+3]))) for i in range(0, len(row)-2,2)]
        x,y,w,h = cv2.boundingRect(np.array([points]))
        
        rectPoints = (x, y, w, h)
        muchtRects.append(rectPoints)
    return muchtRects


def dataModelOpenface(my_data):
    modelRects = []
    dl = AlignDlib("/home/yomna/GraduationProject/openface/models/dlib/shape_predictor_68_face_landmarks.dat")
    for j in range(len(my_data)):
        img = cv2.imread('/home/yomna/GraduationProject/openface/muct-master/jpg'+ str((j % 5) + 1) + '/' + my_data[j][0] + '.jpg')
        rect = dl.getAllFaceBoundingBoxes(img)
        rectObject = rect[0]
        rectPoints = (rectObject.left(), rectObject.top(), rectObject.right(),rectObject.bottom())
        modelRects.append(rectPoints)
    return modelRects


def compareRects(muchtRects,modelRects):
    print muchtRects[0], modelRects[0]
    distance = 0 
    for i in range(len(muchtRects)):
        ### calculating distance
        ## real points 
        rx = muchtRects[i][0]
        ry = muchtRects[i][1]
        rw = muchtRects[i][2]
        rh = muchtRects[i][3]
        ## predicted points
        px = modelRects[i][0]
        py = modelRects[i][1]
        pw = modelRects[i][2]
        ph = modelRects[i][3]
        rP1 = (rx, ry) #real first point
        rP2 = (rx+rw, ry) #real  point
        rP3 = (rx, ry+rh) #real  point
        rP4 = (rx+rw, ry+rh) #real  point

        ## for openFace model
        pP1 = (px, py) #predicted  point
        pP2 = (pw, py) #predicted  point
        pP3 = (px, ph) #predicted  point
        pP4 = (pw, ph) #predicted  point

        ## uncomment the folllowing point for any model except openface
        """pP1 = (px, py) #pridected first point
                                pP2 = (px+pw, py) #pridected first point
                                pP3 = (px, py+ph) #pridected first point
                                pP4 = (px+pw, py+ph) #pridected first point"""
        
        distance += math.sqrt( math.pow(rP1[0]-pP1[0], 2) + math.pow(rP2[0]-pP2[0], 2) + math.pow(rP3[0]-pP3[0], 2) + math.pow(rP4[0]-pP4[0], 2) +
                     math.pow(rP1[1]-pP1[1], 2) + math.pow(rP2[1]-pP2[1], 2) + math.pow(rP3[1]-pP3[1], 2) + math.pow(rP4[1]-pP4[1], 2))/2
    
    print "Error between Real and Predicted is " + str(distance)
       

my_data = genfromtxt('/home/yomna/GraduationProject/openface/muct-master/muct-landmarks/muct76-opencv.csv'
    , delimiter=',', dtype =None)[1:]
print "Loaded"
compareRects(muchtDatasetRects(my_data), dataModelOpenface(my_data))
