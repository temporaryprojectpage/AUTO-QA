import datetime
import json
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../../Data/train/argoverse-tracking',dest='root_dir')
parser.add_argument('--split', default='train')
args = parser.parse_args()



"""
Create Scene file corresponding to each log using label file.
For each log, total scenes are equal to total lidar samples in
log file. All scenes correspoding to same log are kept in same 
scene file. Later all scenes from all log are merged into one. 
"""

map_annotate={   'VEHICLE': 'vehicle',
                 'PEDESTRIAN': 'pedestrian',
                 'ON_ROAD_OBSTACLE': 'obstacle',
                 'LARGE_VEHICLE': 'large_vehicle',
                 'BICYCLE': 'bicycle',
                 'BICYCLIST': 'bicyclist',
                 'BUS': 'bus',
                 'OTHER_MOVER': 'other_mover',
                 'TRAILER': 'trailer',
                 'MOTORCYCLIST': 'motorcyclist',
                 'MOPED': 'moped',
                 'MOTORCYCLE': 'motorcycle',
                 'STROLLER': 'stroller',
                 'EMERGENCY_VEHICLE': 'emergency_vehicle',
                 'ANIMAL': 'animal',
                 'WHEELCHAIR': 'wheelchair',
                 'SCHOOL_BUS': 'school_bus'
             }

def create_scene(log_path):
    date=str(datetime.date.today())
    version="1.0"
    split="train"
    label={
        "info":{"date":date,"version":version,"split":split},
        "scenes":[]
    }
    scenes=[]
    #open label.json file containing all annotation for a given argo scene
    with open(log_path+'/'+'label.json') as json_file:
        file=json.load(json_file)
    
    #read json file
    for frame in file:
        tempdic={}
        tempdic["video_filename"]=frame['context_name']
        tempdic['lidar_index']=frame['lidar_index']
        tempdic["directions"]={"ring front center": [],
                               "ring front left":[],
                                "ring front right":[],
                                "ring rear right":[],
                                "ring rear left":[],
                                "ring side right":[]}
        tempdic["relationships"]={"center":[],
                                  "front left":[],
                                  "front right":[],
                                  "side left":[],
                                  "side right":[],
                                  "rear right":[],
                                  "rear left":[]}
        tempdic["objects"]=[]
        mapname={"ring_front_center":"center", 
                 "ring_front_right":"front right",
                 "ring_front_left":"front left",
                 "ring_side_left":"side left",
                 "ring_side_right":"side right",
                 "ring_rear_left":"rear left",
                 "ring_rear_right":"rear right"}
        obj_counter=1
        for key,value in mapname.items():
            tempdic['relationships'][value].append([])
            
        description={ "shape":"my",
                     "coordinates":[0,0,0,0],
                     "color":'red',
                     'speed':"",
                     'status':'',
                     "collisions": [],
                     'id':"id_of_object"
                    }
        tempdic["objects"].append(description)
        for direction in frame['direction']:
            disha=direction['name']
            for objs in (direction['object']):
                for key,value in mapname.items():
                    tempdic['relationships'][value].append([])
                tempdic['relationships'][mapname[disha]][0].append(obj_counter)
                obj_counter=obj_counter+1
                box=objs['3d coordinates']
                description={ "shape":map_annotate[objs['type']],
                             "coordinates":box,
                             "color":'red',
                             'speed':"",
                             'status':'',
                             "collisions": []
                            }
                tempdic["objects"].append(description)
#                 stats=stats+1
        scenes.append(tempdic)
    label['scenes']=scenes
    return label



if __name__ == '__main__':
    out_dir=os.path.join('../output',args.split+'_scenes')
    if(not os.path.isdir(out_dir)):
        os.makedirs(out_dir)

    for log_id in os.listdir(args.root_dir):
        print('='*70)
        log_path=os.path.join(args.root_dir,log_id)
        label=create_scene(log_path)
        print('generating scene file for ',(log_id))
        with open(os.path.join(out_dir,'scene' + '(' + log_id + ')' + '.json'),'w') as outfile:
            json.dump(label,outfile,indent=4)
        
    print('='*70)
    print()
    print("Successfully completed scene file generation")


