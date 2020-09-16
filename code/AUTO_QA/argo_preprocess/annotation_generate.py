import argparse
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.calibration import project_lidar_to_img,proj_cam_to_uv
from argoverse.utils.json_utils import read_json_file
import copy
import numpy as np
import json
import os



"""
Create label file corresponding to each log file in lidar dataset.This
is done by projecting 3D annotation into images using argovere library.
Also annotation are stored direction wise in a sepearate label.json file
which is stored in the same dataset folder.
"""



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../../Data/train/argoverse-tracking',dest='root_dir')
args = parser.parse_args()




argoverse_loader = ArgoverseTrackingLoader(args.root_dir)
print('Total number of logs:',len(argoverse_loader))
argoverse_loader.print_all()



camera_name=['ring_front_center',
             'ring_side_left',
             'ring_side_right',
             'ring_front_left',
             'ring_front_right',
             'ring_rear_right',
            'ring_rear_left']

def generate_annotation(annotation,argoverse_data,log_id,distance_range=20):
    for idx in range(len(argoverse_data.image_list_sync['ring_front_center'])):#just to find length
        # print(idx)
        dic={}
        dic['context_name']=str(log_id)
        dic['frame_name']=str(argoverse_data.get_image_sync(idx=0,camera=camera_name[0],load=False)).split('/')[-1]
        dic["direction"]=[{"name":"ring_front_center","object":[]},
                                {"name":"ring_front_left","object":[]},
                                {"name":"ring_front_right","object":[]},
                                {"name":"ring_side_left","object":[]},
                                {"name":"ring_side_right","object":[]},
                                {"name":"ring_rear_left","object":[]},
                                {"name":"ring_rear_right","object":[]}
                                ]
        dic['lidar_index']=idx
        for camera in camera_name:
            img = argoverse_data.get_image_sync(idx, camera=camera)
            objects = argoverse_data.get_label_object(idx)
            calib = argoverse_data.get_calibration(camera=camera)
            for obj in objects:
                box3d_pts_3d = obj.as_3d_bbox()
                cx=box3d_pts_3d.mean(axis=0)[0]
                cy=box3d_pts_3d.mean(axis=0)[1]
                dist=np.sqrt(cx**2+cy**2)
                if dist>distance_range:
                    continue
                calib_fpath = os.path.join(args.root_dir,log_id,'vehicle_calibration_info.json')
                calib_data = read_json_file(calib_fpath)
                uv_cam = calib.project_ego_to_cam(box3d_pts_3d)
                uv, uv_cam_T, valid_pts_bool, camera_config = proj_cam_to_uv(uv_cam, copy.deepcopy(calib.camera_config))
                if any(valid_pts_bool):  #all for object completely lie inside the frame
                    objdic={}
                    objdic['type']=obj.label_class
                    objdic['3d coordinates']=box3d_pts_3d.tolist()
                    dic['direction'][camera_name.index(camera)]["object"].append(objdic)
        annotation.append(dic)



if __name__ == '__main__':
    for log_id in argoverse_loader.log_list:
        print('Creating label file for ',format(log_id))
        annotation=list()
        argoverse_data=argoverse_loader.get(log_id)
        generate_annotation(annotation,argoverse_data,log_id)
        with open(os.path.join(args.root_dir,log_id,'label.json'),'w') as f:
            json.dump(annotation,f,indent=4)
        print('Done for ',format(log_id))
    print('all label successfully generated')




