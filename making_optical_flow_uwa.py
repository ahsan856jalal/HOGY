import os,sys
from os.path import join, isfile
import numpy as np
from pylab import *
from PIL import Image
import cv2
from scipy.misc import imresize
import glob
import matplotlib.pyplot as plt
from scipy.misc import imresize

import numpy as np
num = np.zeros(17)
fish_list=['Abudefduf_bengalensis',
           'Carangoides_fulvoguttatus',
           'Choerodon_cyanodus',
           'Choerodon_rubescens',
           'Coris_auricularis',
           'Lethrinus_atkinsoni',
           'Lethrinus_nebulosus',
           'Lethrinus_sp',
           'Lutjanus_carponotatus',
           'Pagrus_auratus',
           'Pentapodus_emeryii',
           'Pentapodus_porosus',
           'Plectropomus_leopardus',
           'Scarus_ghobban',
           'Scombridae_spp',
           'Thalassoma_lunare',
           'Other_Other']


main_dir='/home/user/uwa'
saving_dir='/home/user/uwa/save_dir'


data_txt_dir=open('/home/user/uwa/paths.txt')
data_txt_files=data_txt_dir.readlines()
data_txt_dir.close()
missed_vids=[]
vid_counter=0
for video_abs_path in data_txt_files:
    print('video {}/{} in process'.format(vid_counter,len(data_txt_files)))
    vid_counter+=1
    abs_path=video_abs_path.rstrip()
    vid_path=abs_path.rsplit('/', 1)[0]
    path_split=abs_path.split('/')
    subfolder=path_split[-2]
    family_genus_specie=path_split[-3]
    mainfolder=path_split[-4]
    aa=open(abs_path)
    a=aa.readlines()
    a1=a[0].rstrip()
    a2=a[1].rstrip()
    params_name=a1.split('\t')
    params_value=a2.split('\t')
    fish_name=params_value[13]+'_'+params_value[14]
    if(fish_name=='Lethrinus_sp.'):
        fish_name='Lethrinus_sp'
    if(fish_name=='Scomber_spp'):
        fish_name='Scombridae_spp'
    frame_number=int(params_value[22])
    lx1=int(float(params_value[23]))
    ly1=int(float(params_value[24]))
    lx2=int(float(params_value[25]))
    ly2=int(float(params_value[26]))
    rx1=int(float(params_value[27]))
    ry1=int(float(params_value[28]))
    rx2=int(float(params_value[29]))
    ry2=int(float(params_value[30]))
    if(lx1>lx2):
        tmp=lx2
        lx2=lx1
        lx1=tmp
    if(rx1>rx2):
        tmp=rx2
        rx2=rx1
        rx1=tmp
    if(ly1>ly2):
        tmp=ly2
        ly2=ly1
        ly1=tmp
    if(ry1>ry2):
        tmp=ry2
        ry2=ry1
        ry1=tmp
   
   # now reading left and right frame
    left_vid=join(vid_path,'Left.avi')
    if left_vid:
        cap=cv2.VideoCapture(left_vid)
        # vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2
        # vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2
        counter=0

        cap.set(1, frame_number-1)
        ret, frame1 = cap.read()
        img_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        frame1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)    
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        aa=0
        kernel = np.ones((7,7),np.uint8)
        cap.set(1, frame_number)
        ret, frame2 = cap.read()
        img_yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        frame2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#            frame2=imresize(frame2,[640,640])
        next1 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next1, None, 0.95, 10, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        if not os.path.exists(join(saving_dir,fish_name)):
                    os.makedirs(join(saving_dir,fish_name))
        cv2.imwrite(join(saving_dir,fish_name)+'/'+mainfolder+'__'+subfolder+'__'+fish_name+'name'+'.png',opening)

    else:
        print('video   {}  is missing'.format(left_vid))
        missed_vids.append(left_vid)
