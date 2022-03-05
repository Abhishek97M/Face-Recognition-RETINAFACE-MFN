import cv2
from Algorithms.centroidTracker import CentroidTracker
from PIL import Image
import imutils
import numpy as np
import time
from time import sleep, ctime
import os
from datetime import date, datetime
#from matplotlib import pyplot as plt
class VideoCamera(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./primary_face_detector/haarcascade_frontalface_default.xml')
        self.video = cv2.VideoCapture(0)
        self.ct = CentroidTracker()
        self.H = None
        self.W = None
        self.ids = -1

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if (success == True):
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            rects=[]
            for (x,y,w,h) in faces:
                
                if x>180 and y >60 and x+w <430 and y+h<320:
                  if w>150 and h>150:
                    rects.append(np.array([x,y,x+w,y+h]))
                    cv2.rectangle(frame, (x, y), (x+w, y+h),
				(0, 255, 0), 2)
                    
            fcs = []
            objects = self.ct.update(rects)
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                #cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                if objectID > self.ids:
                        self.ids = objectID
                        fcs.append(frame)
            cv2.circle(frame, (320,181), 150, (255,0,0), 1)
            return frame,fcs

    def get_one(self):
        success, frame = self.video.read()
        if (success == True):
            frame = frame[20:387,148:909]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                return frame,1
            else:
                return frame,0
