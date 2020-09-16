import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import argparse, os, torch
import torch.nn as nn
import torch.nn.functional as F
from aqa.data import argo_collate
import torch.optim as optim
from aqa.data import ArgoDataLoader


from aqa.futils import get_answer_classes, load_vocab, correct_pred_count
from aqa.futils import load_weights
from tqdm import tqdm
import gc
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")





#######seperate models###########################################3
from lidar_models.san_lidar import SAN_LIDAR
from lidar_models.mcb_lidar import MCB_LIDAR
from lidar_models.mfb_lidar import MFB_LIDAR
from lidar_models.mlb_lidar import MLB_LIDAR,MUTAN_LIDAR
from aqa.model import LSTM_BASIC,CNN_LSTM,SAN,MCB,DAN,MFB,LIDAR_MODEL,MUTAN,MLB



torch.manual_seed(7)
parser = argparse.ArgumentParser()



#######################paths for various files needed#####################################
parser.add_argument('--input_base', default='../../output/processed')
parser.add_argument('--image_features',default='features.h5')
parser.add_argument('--test_encodings', default='test_questions.h5')
parser.add_argument('--vocab', default='vocab_test.json')

parser.add_argument('--model_name', default='../../output')  ##used for weights loading




######################### Model Parameters #################################################
parser.add_argument('--test_batch_size', default=2, type=int)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--encoder_type', default='gru', choices=['lstm', 'gru'])
parser.add_argument('--model_dir', default='../../output/new') ##directory containing all saved model
parser.add_argument('--test_num_workers', default=0, type=int)
parser.add_argument('--fusion_type', dest='method',default='concat',help='Fusion strategy concat/hierarchical')
parser.add_argument('--grouping',default='single_scale',help='Grouping method for point cloud')





parser.add_argument('--model_type', default='LSTM_BASIC', choices=['SAN','MCB','MFB','MLB','MUTAN','DAN'],
                    help='Change the model to different model')

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

   
    
    test_loader_kwargs = dict(
        question_h5=os.path.join(args.input_base, args.test_encodings),
        vocab=os.path.join(args.input_base, args.vocab),
        image_feature_h5=os.path.join(args.input_base, args.image_features),
        batch_size= args.test_batch_size,
        lidar_feature_h5='../Data/test/argoverse-tracking',
        load_lidar=False,
        num_workers=args.test_num_workers,
        collate_fn=argo_collate
    )

    with ArgoDataLoader(**test_loader_kwargs) as test_loader:
        test_loop(args,test_loader)


def test_loop(args,test_loader):

    question_feat_size = 1024
    image_feature_size=2048
    lidar_feature_size=1024

    if args.model_type == 'LSTM_BASIC':
        model = LSTM_BASIC(args, question_feat_size, image_feature_size,lidar_feature_size,num_classes=34, encoder=args.encoder_type)
    elif args.model_type == 'CNN_LSTM':
        model=CNN_LSTM(args,question_feat_size, image_feature_size, num_classes=34, qa=None, encoder=args.encoder_type)
    elif args.model_type=='SAN':
        question_feat_size=512
        model=SAN(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method)
    elif args.model_type=='MCB':
        question_feat_size=512
        model=MCB(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method)
    elif args.model_type=='DAN':
        question_feat_size=512
        model=DAN(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method)
    elif args.model_type=='MFB':
        question_feat_size=512
        image_feature_size=512
        model=MFB(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method)
    elif args.model_type=='MLB':
        question_feat_size=1024
        image_feature_size=512
        model=MLB(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method)
    elif args.model_type=='MUTAN':
        question_feat_size=1024
        image_feature_size=512
        model=MUTAN(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method)
    elif args.model_type=='MCB_LIDAR':
        question_feat_size=512
        model=MCB_LIDAR(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,grouping=args.grouping)
    elif args.model_type=='MFB_LIDAR':
        question_feat_size=512
        image_feature_size=512
        model=MFB_LIDAR(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,grouping=args.grouping)
    elif args.model_type=='SAN_LIDAR':
        question_feat_size=512
        model=SAN_LIDAR(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,grouping=args.grouping)
    elif args.model_type=='MLB_LIDAR':
        question_feat_size=1024
        image_feature_size=512
        model=MLB_LIDAR(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,grouping=args.grouping)
    elif args.model_type=='MUTAN_LIDAR':
        question_feat_size=1024
        image_feature_size=512
        model=MUTAN_LIDAR(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,grouping=args.grouping)
    elif args.model_type=='LIDAR_MODEL':
        model=LIDAR_MODEL(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method,grouping=args.grouping)
    
    else:
        raise NotImplementedError('Given Module is not implemented')
    
    
    
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
    total_corrects = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
                question, image_feature,ques_lengths,point_set,answer=data
                question = question.to(device=args.device)
                ques_lengths = ques_lengths.to(device=args.device)
                image_feature=image_feature.to(device=args.device)
                point_set=point_set.to(device=args.device)
                
                
                pred = model(question, image_feature,ques_lengths,point_set)
                
                predictions = pred.detach().cpu()
                correct_count_vec, correct_count = correct_pred_count(predictions, answer)
                total_corrects += correct_count.item()
                batch_accuracy = correct_count.item() / question.size(0)
                gc.collect()


        total_samples = test_loader.get_dset_size()
        accuracy = total_corrects / total_samples
        print(" Accuracy: %.3f \n Total samples: %d \n "
              % (accuracy, total_samples),flush=True)
if __name__=='__main__':
    args=parser.parse_args()
    main(args)

