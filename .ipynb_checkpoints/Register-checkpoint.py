import cv2
import os
from utils.arcface_recognition import  face_registered_projector
import argparse
from utils import face_model
vid = cv2.VideoCapture('rtsp://admin:jetson123@192.168.20.128:554/Streaming/Channels/101')
#img = cv2.imread("")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face embedding and comparison')
    parser.add_argument('--image-size', default='112,112', help='') #follow sphere face & cosin face format
    #parser.add_argument('--model', default='./models/model-mobileface-4/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id,(-1) for CPU')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--image_mode', default=False, action="store_true",help='Image detection mode')
    parser.add_argument('--regist', default=False, action="store_true",help='to regist face or to compare face')
    args = parser.parse_args()
    model = face_model.FaceModel(args)
    cwd = os.getcwd()
    #vid = cv2.VideoCapture(0)
    #img = cv2.imread("C:/Users/achoudhary/Desktop/CYBERDYNE/CYBERDYNE-FS/WIN_20210920_15_51_30_Pro.jpg")
    #face_registered_projector(model,img)
    while(True):
        return_value, frame = vid.read()
        
        face_registered_projector(model,frame)
