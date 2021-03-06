import cv2
import numpy as np
from matplotlib import pyplot as plt
import os,sys
from os.path import join, isfile
import numpy as np
from pylab import *
from PIL import Image
import cv2
from scipy.misc import imresize
import glob
import matplotlib.pyplot as plt
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
saving_dir='/home/user/save_dir'

# reading absolute paths of all Data.txt files in main directory
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
    if(fish_name=='_spp'):
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
   
    left_vid=join(vid_path,'Left.avi')
    if left_vid:
        cap=cv2.VideoCapture(left_vid)
        counter=0
        success,image = cap.read()
        while success:
            if counter==frame_number:
                template_left=cv2.imread(join(vid_path,'left-fish.jpg'))
                template_left=cv2.cvtColor(template_left,cv2.COLOR_BGR2GRAY)
                source_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                img_width,img_height=source_image.shape[::-1]
                w, h = template_left.shape[::-1]
                res = cv2.matchTemplate(source_image,template_left,eval('cv2.TM_SQDIFF'))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = min_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                img2=image.copy()
                if not os.path.exists(join(saving_dir,fish_name)):
                    os.makedirs(join(saving_dir,fish_name))
                cv2.imwrite(join(saving_dir,fish_name)+'/'+mainfolder+'__'+subfolder+'__'+fish_name+'__left'+'.png',image)
                x=top_left[0]
                y=top_left[1]
                w=bottom_right[0]-top_left[0]
                h=bottom_right[1]-top_left[1]
                x = (x+w/2.0) / img_width
                y = (y+h/2.0) / img_height
                w = float(w) / img_width
                h = float(h) / img_height
                if fish_name in fish_list:
                    fish_specie=fish_list.index(fish_name)
                    tmp = [fish_specie, x, y, w, h]
                    xml_content = ""
                    xml_content += "%d %f %f %f %f\n" % (tmp[0], tmp[1], tmp[2], tmp[3], tmp[4])
                    f = open(join(saving_dir,fish_name)+'/'+mainfolder+'__'+subfolder+'__'+fish_name+'__left'+'.txt', "w")
                    f.write(xml_content)
                    f.close()
                else:
                    fish_specie=fish_list.index('Other_Other')
                    tmp = [fish_specie, x, y, w, h]
                    xml_content = ""
                    xml_content += "%d %f %f %f %f\n" % (tmp[0], tmp[1], tmp[2], tmp[3], tmp[4])
                    f = open(join(saving_dir,fish_name)+'/'+mainfolder+'__'+subfolder+'__'+fish_name+'__left'+'.txt', "w")
                    f.write(xml_content)
                    f.close()
                    
                
                
                
                counter+=1
                success,image = cap.read()
                
   
            else:
                counter+=1
                success,image = cap.read()
    else:
        print('video   {}  is missing'.format(left_vid))
        missed_vids.append(left_vid)
    