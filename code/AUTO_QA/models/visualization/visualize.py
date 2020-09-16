import sys
import os
sys.path.insert(0, os.path.abspath('../'))
# print(sys.path)
import argparse, os, torch
import torch.nn as nn
import torch.nn.functional as F
from data import argo_collate
import torch.optim as optim
from data import ArgoDataLoader


from aqa.futils import get_answer_classes, load_vocab
from aqa.futils import load_weights
from tqdm import tqdm
import gc
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")



###from visualizion folder####################################
from utils import get_ans,get_ques,plot_att


#######seperate models###########################################3
from san import SAN
from mcb import MCB
from mfb import MFB
from dan import DAN
from mlb import MLB,MUTAN



torch.manual_seed(7)
parser = argparse.ArgumentParser()



#######################paths for various files needed#####################################
parser.add_argument('--input_base', default='../../output/processed')
parser.add_argument('--image_features',default='features.h5')
parser.add_argument('--val_encodings', default='val_questions2.h5')
parser.add_argument('--vocab', default='vocab2.json')

parser.add_argument('--model_name', default='../../output')  ##used for weights loading




######################### Model Parameters #################################################
parser.add_argument('--val_batch_size', default=2, type=int)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--encoder_type', default='gru', choices=['lstm', 'gru'])

parser.add_argument('--model_dir', default='../../output/new') ##directory containing all saved model
parser.add_argument('--save_dir', default='../../output/new2')

parser.add_argument('--val_num_workers', default=0, type=int)



parser.add_argument('--model_type', default='LSTM_BASIC', choices=['SAN','MCB','MFB','MLB','MUTAN','DAN'],
                    help='Change the model to different model')

# parser.add_argument('--features_type', default='vgg_14',
#                     choices=['resnet_1', 'vgg_1', 'resnet_14', 'vgg_14', 'resnet_7'])



########################################################################################################
def main(args):
    """
    :type args: argument parser object
    """
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')
    
    
    if not os.path.isdir(args.model_dir):
        print('No such model directory exist')
        return

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    val_loader_kwargs = dict(
        question_h5=os.path.join(args.input_base, args.val_encodings),
        vocab=os.path.join(args.input_base, args.vocab),
        image_feature_h5=os.path.join(args.input_base, args.image_features),
        batch_size= args.val_batch_size,
        lidar_feature_h5='../../Data/train/argoverse-tracking',
        load_lidar=False,
        num_workers=args.val_num_workers,
        collate_fn=argo_collate
    )

    with ArgoDataLoader(**val_loader_kwargs) as val_loader:
        visualize_loop(args,val_loader)


def visualize_loop(args,val_loader):

    image_feature_size=512
    lidar_feature_size=1024

    if args.model_type=='SAN':
        question_feat_size=512
        model=SAN(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method='hierarchical')
    if args.model_type=='MCB':
        question_feat_size=512
        model=MCB(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method='hierarchical')
    if args.model_type=='MFB':
        question_feat_size=512
        # image_feature_size=512
        model=MFB(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method='hierarchical')
    if args.model_type=='MLB':
        question_feat_size=1024
        image_feature_size=512
        model=MLB(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method='hierarchical')
    if args.model_type=='MUTAN':
        question_feat_size=1024
        image_feature_size=512
        model=MUTAN(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method='hierarchical')
    if args.model_type=='DAN':
        question_feat_size=512
        model=DAN(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method='hierarchical')
    
    data = load_weights(args, model, optimizer=None)
    if type(data) == list:
        model, optimizer, start_epoch, loss, accuracy = data
        print("Loaded  weights")
        print("Epoch: %d, loss: %.3f, Accuracy: %.4f " % (start_epoch, loss, accuracy),flush=True)
    else:
        print(" error occured while loading model training freshly")
        model = data
        return


    ###########################################################################multiple GPU use#
    # if torch.cuda.device_count() > 1:
    #     print("Using ", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)


    model.to(device=args.device)
    model.eval()


    import argoverse
    from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
    from argoverse.utils.json_utils import read_json_file
    from argoverse.map_representation.map_api import ArgoverseMap

    vocab = load_vocab(os.path.join(args.input_base, args.vocab))
    argoverse_loader=ArgoverseTrackingLoader('../../../Data/train/argoverse-tracking')

    k=1   
    with torch.no_grad():
        for data in tqdm(val_loader):
            question, image_feature,ques_lengths,point_set,answer,image_name=data
            question = question.to(device=args.device)
            ques_lengths = ques_lengths.to(device=args.device)
            image_feature=image_feature.to(device=args.device)
            point_set=point_set.to(device=args.device)

            pred,wgt,energies=model(question, image_feature,ques_lengths,point_set)


            question=question.cpu().data.numpy()
            answer=answer.cpu().data.numpy()
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred=np.asarray(pred.cpu().data)
            wgt=wgt.cpu().data.numpy()
            energies=energies.squeeze(1).cpu().data.numpy()
            ques_lengths=ques_lengths.cpu().data.numpy()
            pat=re.compile(r'(.*)@(.*)')
            _,keep=np.where([answer==pred])
            temp_batch_size=question.shape[0]
            for b in range(temp_batch_size):
                q=get_ques(question[b],ques_lengths[b],vocab)
                ans=get_ans(answer[b])
                pred_ans=get_ans(pred[b])
                # print(q,ans)
                c=list(re.findall(pat,image_name[b]))[0]
                log_id=c[0]
                idx=int(c[1])
                print(k)
                argoverse_data=argoverse_loader.get(log_id)
                if args.model_type=='SAN':
                    plot_att(argoverse_data,idx,wgt[b,:,1,:],energies[b],q,ans,args.save_dir,k,pred_ans)
                if args.model_type=='MCB':
                    plot_att(argoverse_data,idx,wgt[b],energies[b],q,ans,args.save_dir,k,pred_ans)
                if args.model_type=='MFB':
                    plot_att(argoverse_data,idx,wgt[b,:,:,1],energies[b],q,ans,args.save_dir,k,pred_ans)
                if args.model_type=='MLB':
                    plot_att(argoverse_data,idx,wgt[b,:,3,:],energies[b],q,ans,args.save_dir,k,pred_ans)
                if args.model_type=='MUTAN':#only two glimpses
                    plot_att(argoverse_data,idx,wgt[b,:,1,:],energies[b],q,ans,args.save_dir,k,pred_ans)
                if args.model_type=='DAN':#only two memory
                    plot_att(argoverse_data,idx,wgt[b,:,1,:],energies[b],q,ans,args.save_dir,k,pred_ans)

                k=k+1


if __name__=='__main__':
    args=parser.parse_args()
    main(args)

