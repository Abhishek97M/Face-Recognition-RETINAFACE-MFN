from utils import face_model
import argparse
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw, ImageFont 
import glob
import os
import time
from timeit import default_timer as timer
registered_folder = 'Face_data'


def face_registered_projector(model,image):
        cwd = os.getcwd()
        faces,points,bbox = model.get_input(image)
        for i in range(faces.shape[0]):
            face = faces[i]
            f1 = model.get_feature(face)
            margin = 44
            x1 = int(np.maximum(np.floor(bbox[i][0]-margin/2), 0) )
            y1 = int(np.maximum(np.floor(bbox[i][1]-margin/2), 0) )
            x2 = int(np.minimum(np.floor(bbox[i][2]+margin/2), image.shape[1]) )
            y2 = int(np.minimum(np.floor(bbox[i][3]+margin/2), image.shape[0]) )
            cv2.imshow("Detection",image[y1:y2, x1:x2])
            cv2.waitKey(0)
            x=input("input name for the user : ")
            y=input("input employee id : ")
            print(x)
            npy_name = x+"#"+y+'.npy'
            jpg_name = x+"#"+y+'.jpg'
            np.save(os.path.join(cwd,registered_folder,npy_name),f1)
            cv2.imencode('.jpg', image[y1:y2, x1:x2])[1].tofile(os.path.join(cwd,registered_folder,jpg_name))
            cv2.destroyAllWindows()
       


def registered_face_loader():
    cwd = os.getcwd()
    registed_npy_list = glob.glob(os.path.join(cwd,registered_folder,'*.npy'))
    registered_feature = []
    cat = []
    for npy in registed_npy_list:
        f1 = np.load(npy)
        cat.append(npy.split('\\')[-1].split('.npy')[0])
        registered_feature.append(f1)
    if registed_npy_list==[]:
        print('there is no .npy file in registed face folder')
    else:
        print('load registed %d faces with %d dimensions'%(len(registered_feature),registered_feature[0].shape[0]))
    print('registed names:')
    print(cat)
    return registered_feature,cat



def face_comparison(img,registered_feature,cat,model,threshold = 0.45):
    gtins = time.time() 
    faces,points,bbox = model.get_input(img)
    #print(time.time()-gtins,"fd")
    img_PIL = img
    all_cat,all_sim = [],[]
    tot_names=[]
    if (len(faces)==0 or len(points)==0 or len(bbox)==0):
      img = img
    
    else:
        tot_names=[]
        for i in range(faces.shape[0]):
            
            face = faces[i]
            gtfes = time.time() 
            f2 = model.get_feature(face)
            #print(time.time()-gtfes,"fr")
            sim_record = []
            for j in range(len(registered_feature)):
                sim_record.append(np.dot(registered_feature[j], f2.T))
            most_sim_ind = sim_record.index(max(sim_record))                
            margin = 44
            x1 = int(np.maximum(np.floor(bbox[i][0]-margin/2), 0) )
            y1 = int(np.maximum(np.floor(bbox[i][1]-margin/2), 0) )
            x2 = int(np.minimum(np.floor(bbox[i][2]+margin/2), img.shape[1]) )
            y2 = int(np.minimum(np.floor(bbox[i][3]+margin/2), img.shape[0]) )
            
            #draw = ImageDraw.Draw(img_PIL)
            if sim_record[most_sim_ind]>=threshold:
                text = cat[most_sim_ind].split('/')[-1]+','+str(np.round(sim_record[most_sim_ind],3))
                tot_names.append(cat[most_sim_ind])
                cv2.rectangle(img_PIL, (x1, y2+i), (x2-i, y1-i),
				(0, 0, 255), 2)
                cv2.putText(img_PIL, text, (x1, y1-i),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                
                      
            else:
                
                cv2.rectangle(img_PIL, (x1, y2+i), (x2-i, y1-i),(0,0,255), 2)
            
            #del draw
            all_cat.append(cat)
            all_sim.append(sim_record)
    return all_cat,all_sim,img_PIL,tot_names


def face_comparison_video(registered_feature,cat,model,input_path,threshold = 0.45):
    if input_path=='':
	    vid = cv2.VideoCapture("rtsp://admin:jetson123@192.168.20.128:554/Streaming/Channels/101")
    else:
        vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    countn=0
    while True:
        return_value, frame = vid.read()
        if countn==0 :
            if return_value == True:
                all_cat,all_sim,image,tot_names = face_comparison(frame,registered_feature,cat,model,threshold = 0.45)
                #img_OpenCV = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
                result = np.asarray(image)
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                #cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #            fontScale=0.50, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)
            else:
                print("Frame Break")
                face_comparison_video(registered_feature,cat,model,input_path,threshold = 0.45)
            countn+=1
        else:
            countn=0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()





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
    print("RiGHT")
    cwd = os.getcwd()
    while(True):
        regist_mode_input = input('(For recognition press 0 / For new entries press 1):')
        if regist_mode_input == "0":
            registered_feature,cat = registered_face_loader()
            face_comparison_video(registered_feature,cat,model,"rtsp://admin:jetson123@192.168.20.128:554/Streaming/Channels/101",threshold = 0.45)
        elif regist_mode_input == "1":
            vid = cv2.VideoCapture(0)
            return_value, frame = vid.read()
            face_registered_projector(model,frame)
