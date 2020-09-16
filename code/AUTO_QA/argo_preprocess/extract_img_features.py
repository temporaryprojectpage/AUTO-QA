import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import argparse, json, os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from PIL import Image
import h5py
import numpy as np
from torchvision.transforms import transforms
import torchvision.models as models
import copy
from tqdm import tqdm


import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.calibration import project_lidar_to_img,proj_cam_to_uv
from argoverse.utils.json_utils import read_json_file


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='../../Data/train/argoverse-tracking')
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--num_frames', default=7, type=int, help='number of frames in video')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--output_base', default='../output/processed')
parser.add_argument('--model_type', default='resnet152', help='pretrained model features to use')
parser.add_argument('--split', default='train')
parser.add_argument('--prefix' , default='ARGO')
parser.add_argument('--img_size',default=224,type=int,help='img size to be used for given model')


class ArgoDataset(Dataset):
    def __init__(self,args,transform=None):
        self.args=args
        self.transform=transform
        self.argoverse_loader = ArgoverseTrackingLoader(args.root_dir)
        self.image_list=[]
        for log_id in self.argoverse_loader.log_list:
            argoverse_data=self.argoverse_loader.get(log_id)
            for idx in range(len(argoverse_data.image_list_sync['ring_front_center'])):
                self.image_list.append(str(log_id)+'@'+str(idx))
        self.camera_name=['ring_front_center',
             'ring_side_left',
             'ring_side_right',
             'ring_front_left',
             'ring_front_right',
             'ring_rear_right',
            'ring_rear_left']

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
        lidar_index=int(str(self.image_list[index]).split('@')[1])
        log_id=str(self.image_list[index]).split('@')[0]
        argoverse_data=self.argoverse_loader.get(log_id)
        image_7=[]
        for camera in self.camera_name:
            img = argoverse_data.get_image_sync(lidar_index, camera=camera)
            img=Image.fromarray(img)
            img =self.transform(img) if self.transform is not None else img
            image_7.append(img)
        sample={'image_7':image_7,'video_idxs':lidar_index,'image_names':log_id}
        return sample


def collate_video(batch):
    image_names = []
    video_idxs = []
    image_batch=[]

    # b,7,2048 for resnet 152
    for sample in batch:
        image_names.append(sample['image_names'])
        video_idxs.append(sample['video_idxs'])
        image_batch += sample['image_7']   #extend list
    #print(image_batch.shape)  #b*7,2048
    return image_names, video_idxs, default_collate(image_batch)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x





def main(args):
    
    transform = transforms.Compose([transforms.Resize((args.img_size,args.img_size)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])])
    dset_train = ArgoDataset(args,transform)
    train_loader =DataLoader(dset_train, batch_size=args.batch_size,shuffle=False,collate_fn=collate_video)
    # dataloader_iterator = iter(train_loader)
    # X= next(dataloader_iterator)
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')


    if args.model_type=='resnet152':
        model = models.resnet152(pretrained=True)
        for par in model.parameters():
            par.requires_grad = False
        # model.layer4.register_forward_hook(hook)
        model.fc = Identity()
        
    if args.model_type=='resnet101':
        resnet= models.resnet101(pretrained=True)
        model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
    if args.model_type=='vgg16':
        vgg16=models.vgg16(pretrained=True)
        model=vgg16.features

    # Enable multi-GPU execution if needed.
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
   
    model.to(device=args.device)
    model.eval()

    filename = os.path.join(args.output_base,args.model_type + '_' + args.split + '_features.h5')
    hf = h5py.File(filename, 'w')

    with torch.no_grad():
        for batch in tqdm(train_loader):
            image_names,video_idxs,images=batch
            images = images.to(device=args.device)
            output = model(images)
            output = output.detach().cpu()
            print(output.shape)
            for i in range(output.size(0)//7):
                start = i * args.num_frames
                end = (i + 1) * args.num_frames
                name=str(image_names[i])+'@'+str(video_idxs[i])
                hf.create_dataset(name=name, data=output[start:end])
    hf.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)




    
