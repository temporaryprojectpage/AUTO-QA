import sys
import os.path
import math
import torch.nn.functional as F
import json
import scipy.misc
import numpy as np
import cv2
from aqa.futils import get_clevr_classes
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from skimage import transform, filters
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy
import random
import PIL
from PIL import Image





# camera_name=['ring_front_center', 1
#              'ring_side_left', 3
#              'ring_side_right', 4
#              'ring_front_left', 0
#              'ring_front_right', 2
#              'ring_rear_right', 6
#             'ring_rear_left'] 5

camera_name=['ring_front_center', 
             'ring_side_left', 
             'ring_side_right', 
             'ring_front_left', 
             'ring_front_right', 
             'ring_rear_right', 
            'ring_rear_left'] 



new_camera_name=['ring_front_left',
                 'ring_front_center',
                 'ring_front_right',
                 'ring_side_left',
                 'ring_side_right',
                 'ring_rear_left',
                 'ring_rear_right'
                ]
dic_map={"ring_front_left":"front left",
            "ring_front_center":"center",
            "ring_front_right":"front right",
            "ring_side_left":"side left",
            "ring_side_right":"side right",
            "ring_rear_left":"rear left",
            "ring_rear_right":"rear right"
}
def plot_att(argoverse_data,idx,wgt,energies,ques,ans,save_dir,k,pred_ans):

    wgt1=np.asarray(wgt)
    fig,ax=plt.subplots(7,3,figsize=(10,20))
    order=[3,0,4,1,2,6,5]
    camera_name=['ring_front_center', 
             'ring_side_left', 
             'ring_side_right', 
             'ring_front_left', 
             'ring_front_right', 
             'ring_rear_right', 
            'ring_rear_left'] 
    for i in range(len(order)):
        att=copy.deepcopy(wgt1[order[i]])
        att=att.reshape(14,14)
        img = argoverse_data.get_image_sync(idx, camera=camera_name[order[i]])
        img=Image.fromarray(img)
        img=np.array(img.resize((448,448), PIL.Image.BICUBIC))
        ax[i,0].axis('off')
        ax[i,1].axis('off')
        ax[i,2].axis('off')
        ax[i,0].imshow(np.array(img,np.int32))
        ax[i,1].imshow(get_blend_map_att(img/255.0, np.asarray(att)))
        ax[i,2].imshow(get_blend_map_att(img/255.0, np.asarray(att)),alpha=get_att_level2(energies)[order[i]])
    plt.suptitle("   "+ques+"       "+ans+" "+pred_ans)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    plt.savefig(save_dir+'/'+str(k)+'.png',bbox_inches="tight")




def get_ques(ques,q_len,vocab):
    ques=ques[1:q_len-1]
    q=''
    for i in ques:
        q=q+" "+(vocab['question_idx_to_token'][i])
    return q+" "+'?'


def get_ans(ans):
    classes=get_clevr_classes()
    return classes[ans]


def get_att_level2(cam,ths=0.2):
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam)-np.min(cam))
    # return np.array([i for i in cam])
    return np.array([0.7 if i==1 else ths for i in cam])



def get_p_gradcam(grads_val, target):
    cams = []
    for i in range(grads_val.shape[0]):
        weights = np.mean(grads_val[i], axis = 0)
        cam = np.zeros(target[i].shape[1 : ], dtype = np.float32)

        for k in range(target.shape[0]):
            cam += weights * target[i, k, :, :]
        cams.append(cam)

    return cams


def get_blend_map_gradcam(img, gradcam_map):
        cam = np.maximum(gradcam_map, 0)
        cam = cv2.resize(cam, img.shape[:2])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
#         print(cam.shape)

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        # cv2.imwrite(str(i) + str(j) + "cam.jpg", np.uint8(255 * cam))
        return cam
    
def get_blend_map_att(img, att_map, blur=True, overlap=True):
    att_map -= att_map.min()
    if att_map.max() > 0:
        att_map /= att_map.max()
    att_map = transform.resize(att_map, (img.shape[:2]), order = 3)
    if blur:
        att_map = filters.gaussian(att_map, 0.02*max(img.shape[:2]))
        att_map -= att_map.min()
        att_map /= att_map.max()
    cmap = plt.get_cmap('jet')
    att_map_v = cmap(att_map)
    att_map_v = np.delete(att_map_v, 3, 2)
    if overlap:
        att_map = (1-att_map**0.7).reshape(att_map.shape + (1,))*img + (att_map**0.7).reshape(att_map.shape+(1,)) * att_map_v
        # att_map = img + 0.5 * att_map_v
    return att_map

