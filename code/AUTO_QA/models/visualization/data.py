import numpy as np
import h5py
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from aqa.futils import get_answer_classes, load_vocab



###############argoverse file#############################################
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.json_utils import read_json_file
from argoverse.map_representation.map_api import ArgoverseMap
class ArgoDataset(Dataset):
    def __init__(self, question_h5, image_feature_h5_path, lidar_feature_h5_path,vocab,load_lidar=True,npoint=1024,normal_channel=True,uniform=False,cache_size=15000,drivable_area=False,
                 mode='prefix', image_h5=None, lidar_h5=None, max_samples=None, question_families=None,
                 image_idx_start_from=None):

        #############read whole question_h5 file in memory#############################
        self.all_questions = question_h5['questions'][:]
        self.all_answers = get_answer_classes(question_h5['answers'][:], vocab)
        self.all_image_idxs = question_h5['image_idxs'][:]
        self.all_video_names = (question_h5['video_names'][:]).astype(str)
        self.questions_length = question_h5['question_length'][:]
        self.image_feature_h5 = image_feature_h5_path
        self.load_lidar=load_lidar

        ############for lidar##########################################################
        if self.load_lidar:
            self.argoverse_loader = ArgoverseTrackingLoader(lidar_feature_h5_path)
            self.am = ArgoverseMap()
            self.drivable_area=drivable_area
        ###############################################################################
    def __len__(self):
        print(self.all_questions.shape[0])
        return self.all_questions.shape[0]

    def __getitem__(self, index):
        question = self.all_questions[index]
        question_length = self.questions_length[index]
        answer = self.all_answers[index]
        image_name = self.all_video_names[index] + '@'+str(self.all_image_idxs[index])

        #####image feature ###############################################
        with h5py.File(self.image_feature_h5, 'r') as img_feat_file:
                image_feature = img_feat_file[image_name][:]
        
        ###########lidar feature###########################################
        if self.load_lidar:
            if index in self.cache:
                lidar_pts = self.cache[index]
            else:
                lidar_index=self.all_image_idxs[index]
                log_id=self.all_video_names[index]
                argoverse_data=self.argoverse_loader.get(log_id)

                lidar_pts = argoverse_data.get_lidar(lidar_index,load=True)
                city_to_egovehicle_se3 = argoverse_data.get_pose(lidar_index)
                city_name = argoverse_data.city_name
                

                roi_area_pts = copy.deepcopy(lidar_pts)
                roi_area_pts = city_to_egovehicle_se3.transform_point_cloud(roi_area_pts)  # put into city coords
                roi_area_pts = self.am.remove_non_roi_points(roi_area_pts, city_name)
                roi_area_pts = self.am.remove_ground_surface(roi_area_pts, city_name)
                roi_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(roi_area_pts)



                if self.drivable_area:
                    driveable_area_pts = copy.deepcopy(roi_area_pts)
                    driveable_area_pts = city_to_egovehicle_se3.transform_point_cloud(driveable_area_pts)  # put into city coords
                    driveable_area_pts = self.am.remove_non_driveable_area_points(driveable_area_pts, city_name)
                    driveable_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(driveable_area_pts)
                    point_set=driveable_area_pts
                else:
                    point_set=roi_area_pts
                del lidar_pts

        ############################################################################################################
        else:
            point_set=5
        return (question, image_feature, question_length,point_set,answer,image_name)



class ArgoDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_h5')
        if 'image_feature_h5' not in kwargs:
            raise ValueError('Must give image_feature_h5')

        if 'lidar_feature_h5' not in kwargs:
                raise ValueError('Must give lidar_feature_h5')

        image_feature_h5_path = kwargs.pop('image_feature_h5')
        load_lidar=kwargs.pop('load_lidar')
        lidar_feature_h5_path = kwargs.pop('lidar_feature_h5')
        vocab_path = kwargs.pop('vocab')
        vocab = load_vocab(vocab_path)
        question_h5_path = kwargs.pop('question_h5')
        print('Reading questions from ', question_h5_path)
        with h5py.File(question_h5_path, 'r') as question_h5:
            self.dataset = ArgoDataset(
                question_h5, image_feature_h5_path, lidar_feature_h5_path,vocab=vocab,load_lidar=load_lidar)
        kwargs['collate_fn'] = argo_collate
        super(ArgoDataLoader, self).__init__(self.dataset, **kwargs)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('file closed')

    def get_dset_size(self):
        return self.dataset.__len__()
    
def argo_collate(batch):
    batch.sort(key=lambda x: x[2], reverse=True) 
    transposed = list(zip(*batch))
    question_batch = default_collate(transposed[0])
    image_feature_batch = default_collate(transposed[1])
    ques_len_batch = default_collate(transposed[2])
    point_set_batch= default_collate(transposed[3])
    ans_batch = default_collate(transposed[4])
    image_name=default_collate(transposed[5])
    return (question_batch, image_feature_batch,ques_len_batch,point_set_batch,ans_batch,image_name)