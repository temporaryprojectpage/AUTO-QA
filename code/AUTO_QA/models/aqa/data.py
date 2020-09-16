import numpy as np
import h5py
import torch
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from aqa.futils import get_answer_classes, load_vocab

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


###############argoverse file#############################################
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.json_utils import read_json_file
from argoverse.map_representation.map_api import ArgoverseMap



# question_h5  contains all questions and answers and programs encoded in proper format
# image_feature_h5 contain all image feature

class ArgoDataset(Dataset):
    def __init__(self, question_h5, image_feature_h5_path, lidar_feature_h5_path,vocab,load_lidar=True,
        npoint=1024,normal_channel=True,uniform=False,cache_size=15000,driveable_area=False,
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


        ###########for lidar##########################################################
        if self.load_lidar:
            self.lidar_feature_h5 = lidar_feature_h5_path
            self.argoverse_loader = ArgoverseTrackingLoader(self.lidar_feature_h5)
            self.am = ArgoverseMap()
            self.driveable_area=driveable_area
        #############################################################################

            self.npoints = npoint
            self.uniform = uniform
            self.normal_channel = normal_channel
            self.cache_size = cache_size  # how many data points to cache in memory
            self.cache = {}  # from index to (point_set, cls) tuple


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
        
        if self.load_lidar:
            # if index in self.cache:
            #     point_set = self.cache[index]
            # else:
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



            if self.driveable_area:
                driveable_area_pts = roi_area_pts
                driveable_area_pts = city_to_egovehicle_se3.transform_point_cloud(driveable_area_pts)  # put into city coords
                driveable_area_pts = self.am.remove_non_driveable_area_points(driveable_area_pts, city_name)
                driveable_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(driveable_area_pts)
                point_set=driveable_area_pts
            else:
                point_set=roi_area_pts
            del lidar_pts
            if self.uniform:
                point_set = furthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

                point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

                if not self.normal_channel:
                    point_set = point_set[:, 0:3]
                if len(self.cache) < self.cache_size:
                    self.cache[index] = point_set
                # print(point_set.shape,type(point_set))
                point_set=point_set.T
                
        else:
            point_set=torch.randn(5)
        

        # image

        
        # image here means all 7 image
        # question feature like pretrained bert or word2vec or learned embeddings
        
        return (question, image_feature, question_length,point_set,answer)


class ArgoDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_h5')
        if 'image_feature_h5' not in kwargs:
            raise ValueError('Must give image_feature_h5')

        if 'lidar_feature_h5' not in kwargs:
                raise ValueError('Must give lidar_feature_h5')

        image_feature_h5_path = kwargs.pop('image_feature_h5')

        # self.image_feature_h5 = h5py.File(image_feature_h5_path, 'r')
        # print('Reading image features from ', image_feature_h5_path)

        lidar_feature_h5_path = kwargs.pop('lidar_feature_h5')
        load_lidar=kwargs.pop('load_lidar')
        
        # print('Reading lidar features from ', lidar_feature_h5_path)
        # self.lidar_feature_h5 = h5py.File(lidra_feature_h5_path, 'r')


        # image
        # lidar

        vocab_path = kwargs.pop('vocab')
        vocab = load_vocab(vocab_path)
        question_h5_path = kwargs.pop('question_h5')
        # disable_lidar=kwargs.pop('disable_lidar')
        print('Reading questions from ', question_h5_path)
        with h5py.File(question_h5_path, 'r') as question_h5:
            self.dataset = ArgoDataset(
                question_h5, image_feature_h5_path, lidar_feature_h5_path,vocab=vocab,load_lidar=load_lidar)
        kwargs['collate_fn'] = argo_collate

        # file closed at this point but allquestion are store in variable
        super(ArgoDataLoader, self).__init__(self.dataset, **kwargs)

    # 	def close(self):
    # if self.image_h5 is not None:
    #   self.image_h5.close()
    # if self.feature_h5 is not None:
    #   self.feature_h5.close()

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

    # image
    return (question_batch, image_feature_batch,ques_len_batch,point_set_batch,ans_batch)
