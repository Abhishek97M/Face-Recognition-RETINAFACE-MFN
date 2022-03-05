from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import onnxruntime as rt
from scipy import misc
import sys
import os
import argparse
import numpy as np
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from utils import face_image
from  utils.retinaface import RetinaFace
from utils import face_preprocess
import time
import tensorflow as tf
from data import cfg_mnet
from utils.prior_box import PriorBox
from utils.box_utils import decode, decode_landm, py_cpu_nms

def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def preprocess(img, input_size):
    """Preprocess an image before TRT retinaface inferencing.

    # Args
        img: uint8 numpy array of shape (img_h, img_w, 3)
        input_size: model input size

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    img = np.float32(img)
    img -= (104, 117, 123)
    height, width, _ = img.shape
    long_side = max(height, width)
    img_pad = np.zeros((long_side, long_side, 3), dtype=img.dtype)
    img_pad[0:0+height, 0:0+width] = img
    img = cv2.resize(img_pad, (input_size, input_size))
    img = img.transpose((2, 0, 1))
    return img
def postprocess(loc, conf, landms, priors, cfg, img):
    """Postprocess TensorRT outputs.

    # Args
        loc: [x, y, w, h]
        conf: [not object confidence, object confidence]
        landms: [eye_left.x, eye_left.y, 
                 eye_right.x, eye_right.y,
                 nose.x, nose.y
                 mouth_left.x, mouth_right.y
                 mouth_left.x, mouth_right.y]
        priors: priors boxes
        cfg: model parameter configure
        img: input image

    # Returns
        facePositions, landmarks (after NMS)
    """
    long_side = max(img.shape)
    img_size = cfg['image_size']
    variance = cfg['variance']
    scale = np.ones(4)*img_size
    scale1 = np.ones(10)*img_size
    confidence_threshold = 0.2
    top_k = 50
    nms_threshold = 0.5

    # decode boxes
    boxes = decode(np.squeeze(loc, axis=0), priors, variance)
    boxes = boxes*scale

    # decode landmarks
    landms = decode_landm(np.squeeze(landms, axis=0), priors, variance)
    landms = landms*scale1  

    # ignore low scores
    scores = np.squeeze(conf, axis=0)[:, 1]
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]   

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS 
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    
    # resize
    res = long_side/img_size
    facePositions = (dets[:,:4]*res).astype(int).tolist()
    landmarks = (landms*res).astype(int).tolist()
    return facePositions, landmarks
class FaceModel:
  
 def __init__(self,args):
    self.args = args
    det=0	
    image_size = (112, 112)
    self.model = None
    self.ga_model = None
    self.model_onnx = onnx.load("./trt-models/model.onnx")
    self.engine = backend.prepare(self.model_onnx, device='CUDA:0')
    self.threshold = 1.24
    self.cfg = cfg_mnet
    self.input_size = self.cfg['image_size']
    self.priorbox = PriorBox(self.cfg, (self.input_size, self.input_size))
    self.priors = self.priorbox.forward()
    self.model_detect_onnx = onnx.load("./trt-models/retinaface-320.onnx")
    self.engine_detect = backend.prepare(self.model_detect_onnx, device='CUDA:0')
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    self.image_size = '112,112'
    self.thresh = 0.8
    self.scales = [350, 350]
    self.flip = False
    count = 1
    gpuid = 0
    
 def get_input(self, face_img):
    img_resized = preprocess(face_img, self.input_size)
    inputs = img_resized[np.newaxis, :, :, :]
    inputs =  np.array(inputs, dtype=inputs.dtype, order='C')
    out = self.engine_detect.run(inputs)
    loc=out.output0
    conf= out._2
    landms=out._1
    #print(out.output0)
    ret = postprocess(loc, conf, landms, self.priors, self.cfg, face_img)
    if ret[0] == []:
        all_aligned = []
        points=[]
        bbox=[]
        return all_aligned,points,bbox
    else:
        bbox, points = ret
        #rint(len(bbox))
        for i in range(len(bbox)):
            single_bbox = bbox[i]
            single_points = np.array([[points[i][0],points[i][1]],[points[i][2],points[i][3]],[points[i][4],points[i][5]],[points[i][6],points[i][7]],[points[i][8],points[i][9]]])
            #rint(single_points)
            nimg = face_preprocess.preprocess(face_img, single_bbox, single_points, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            aligned = nimg
            if i ==0:
                all_aligned = aligned.reshape(1,aligned.shape[0],aligned.shape[1],aligned.shape[2])
            else:
                all_aligned = np.concatenate((all_aligned,aligned.reshape(1,aligned.shape[0],aligned.shape[1],aligned.shape[2])),axis=0 )  
        if len(bbox[0]) > 0:
            return all_aligned,points,bbox
        else:
            return [],[],[]  

 def get_feature(self, aligned):
    #print(aligned.shape)
    aligned = np.expand_dims(aligned, axis=0)
    aligned = np.array(aligned, dtype=np.float32)
    #rint(aligned.shape)
    output_data = self.engine.run(aligned)[0]
    embedding = output_data
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding


