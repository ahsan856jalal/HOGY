
import sys,os,glob
from os.path import join, isfile
import numpy as np
from pylab import *
from PIL import Image
import cv2
import dlib
from scipy.misc import imresize
from statistics import mode
from tempfile import TemporaryFile
from collections import Counter
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
#from rgb2gray import rgb2gray
import lxml.etree
import scipy.misc
from natsort import natsorted, ns
import xml.etree.ElementTree as ET
from shutil import copytree
import matplotlib.pyplot as plt
import glob
import os
import PIL
from ctypes import *
import math
import random
bkg_count=0
###################### incorporating darknet.py ######################################################

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = # lib path
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.2, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
specie_list= ["abudefduf vaigiensis",
             "acanthurus nigrofuscus",
             "amphiprion clarkii",
             "chaetodon lununatus",
             "chaetodon speculum",	
             "chaetodon trifascialis",
             "chromis chrysura",
             "dascyllus aruanus",
             "dascyllus reticulatus",
             "hemigumnus malapterus",
             "myripristis kuntee",
             "neoglyphidodon nigroris",
             "pempheris vanicolensis",
             "plectrogly-phidodon dickii",
            "zebrasoma scopas"]    
# add net and meta files
total_gt_count=0
total_det_cunt=0
TP=0
FP=0
# val list and save dir paths
val_lines=val_list.readlines()
tmp1=0
for img_name in val_lines:
    obj_arr = []
    tmp1+=1
    print(tmp1)
    img_name = img_name.rstrip()
    filename=img_name.split('/')
    img_file=filename[-1]
    video_file=filename[-2]
    gt_text1=open(img_name.split('.png')[0]+'.txt')
    gt_text=gt_text1.readlines()
    gt_count=len(gt_text)
    total_gt_count+=gt_count
    img_rgb=cv2.imread(img_name)
 
    r = detect(net, meta, img_name)
    if r:
        for fish_info in r:
            
                fish_name_det=fish_info[0]
                fish_lab_det=specie_list.index(fish_name_det)
                x=(fish_info[2][0])
                y=(fish_info[2][1])
                w=(fish_info[2][2])
                h=(fish_info[2][3])
                xmin_det = int(x - w/2)
                ymin_det = int(y - h/2)
                xmax_det = int(x + w/2)
                ymax_det = int(y + h/2)

                x1 = x / gt_width
                y1 = y / gt_height
                w1 = float(w) / gt_width
                h1 = float(h) / gt_height
                # img=cv2.rectangle(img_rgb,(xmin_det,ymin_det),(xmax_det,ymax_det),(255,12,0),2)
                # cv2.imwrite(join(saving_dir,video_file)+'/'+ "{}_{}.png".format(tmp1,fish_name_det),img)
                tmp = [fish_lab_det, x1, y1, w1, h1]
                obj_arr.append(tmp)
                if(xmax_det > width):
                    xmax_det=width
              
                if(ymax_det > height):
                    ymax_det=height
             
                match_flag=0
                for line_gt in gt_text:
                    line_gt1 = line_gt.rstrip()
                    coords=line_gt1.split(' ')
                    fish_label_gt=int(coords[0])
                    
                    w_gt=round(float(coords[3])*gt_width)
                    h_gt=round(float(coords[4])*gt_height)
                    x_gt=round(float(coords[1])*gt_width)
                    y_gt=round(float(coords[2])*gt_height)
                    x_gt=int(x_gt)
                    y_gt=int(y_gt)
                    h_gt=int(h_gt)
                    w_gt=int(w_gt)
                    xmin_gt = int(x_gt - w_gt/2)
                    ymin_gt = int(y_gt - h_gt/2)
                    xmax_gt = int(x_gt + w_gt/2)
                    ymax_gt = int(y_gt + h_gt/2)
                    if(xmin_gt<0):
                        xmin_gt=0
                    if(ymin_gt<0):
                        ymin_gt=0
                    if(xmax_gt>gt_width):
                        xmax_gt=gt_width
                    if(ymax_gt>gt_height):
                        ymax_gt=gt_height
                    
                    xa=max(xmin_det,xmin_gt)
                    ya=max(ymin_det,ymin_gt)
                    xb=min(xmax_det,xmax_gt)
                    yb=min(ymax_det,ymax_gt)
                    if(xb>xa and yb>ya):
                        match_flag+=1
                        area_inter=(xb-xa+1)*(yb-ya+1)
                        area_gt=(xmax_gt-xmin_gt+1)*(ymax_gt-ymin_gt+1)
                        area_pred=(xmax_det-xmin_det+1)*(ymax_det-ymin_det+1)
                        area_min=min(area_gt,area_pred)
                        area_union=area_pred+area_gt-area_inter
                        if(float(area_inter)/area_min>=0.5): # IOMin\

                           if(fish_lab_det==fish_label_gt):
                               TP+=1
                           else:
                               FP+=1
                       else:
                           FP+=1
        if(match_flag==0):
            FP+=1
               
   xml_content = ""
   for obj in obj_arr:
       xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])
   if not os.path.exists(join(saving_dir,video_file)):
       os.makedirs(join(saving_dir,video_file))
   f = open(join(saving_dir,video_file,img_file).split('.png')[0]+'.txt', "w")
   f.write(xml_content)
   f.close()

