import argparse, os, torch
import torch.nn as nn
from aqa.data import argo_collate
import torch.optim as optim
from aqa.data import ArgoDataLoader
from aqa.futils import correct_pred_count, save_model, load_weights, train_base_masking
from aqa.model import LSTM_BASIC,CNN_LSTM,SAN,MCB,DAN,MFB,LIDAR_MODEL,MUTAN,MLB

import gc

torch.manual_seed(7)
parser = argparse.ArgumentParser()



#######################paths for various files needed#####################################
parser.add_argument('--input_base', default='../output/processed')
parser.add_argument('--image_features',default='features.h5')
parser.add_argument('--train_encodings', default='train_questions.h5')
parser.add_argument('--val_encodings', default='val_questions.h5')
parser.add_argument('--vocab', default='vocab_train.json')
parser.add_argument('--model_name', default='../output')




######################### Model Parameters #################################################
parser.add_argument('--train_batch_size', default=2, type=int)
parser.add_argument('--val_batch_size', default=2, type=int)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--encoder_type', default='gru', choices=['lstm', 'gru'])
parser.add_argument('--num_epochs', default=100, type=int)

parser.add_argument('--model_dir', default='../output/new')
parser.add_argument('--train_num_workers', default=0, type=int)
parser.add_argument('--val_num_workers', default=0, type=int)
parser.add_argument('--lr', default=0.03, type=float)

parser.add_argument('--skip_val', action='store_true')
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--load_lidar', action='store_true', help='Enable Point Cloud loading')
parser.add_argument('--fusion_type', dest='method',default='concat',help='Fusion strategy concat/hierarchical')
parser.add_argument('--grouping',default='single_scale',help='Grouping method for point cloud')




parser.add_argument('--model_type', default='LSTM_BASIC', choices=['LSTM_BASIC','CNN_LSTM','SAN','MCB','DAN','MFB','LIDAR_MODEL','MUTAN','MLB'],
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
        os.makedirs(args.model_dir)
    
    train_loader_kwargs = dict(
        question_h5=os.path.join(args.input_base, args.train_encodings),
        vocab=os.path.join(args.input_base, args.vocab),
        image_feature_h5=os.path.join(args.input_base, args.image_features),
        batch_size=args.train_batch_size,
        lidar_feature_h5='../../Data/train/argoverse-tracking',
        load_lidar=args.load_lidar,
        num_workers=args.train_num_workers,
        collate_fn=argo_collate
    )
    val_loader_kwargs = dict(
        question_h5=os.path.join(args.input_base, args.val_encodings),
        vocab=os.path.join(args.input_base, args.vocab),
        image_feature_h5=os.path.join(args.input_base, args.image_features),
        batch_size= args.val_batch_size,
        lidar_feature_h5='../../Data/train/argoverse-tracking', #it doesnot which path is used because images are not taken from directory
        load_lidar=args.load_lidar,
        num_workers=args.val_num_workers,
        collate_fn=argo_collate
    )
    with ArgoDataLoader(**train_loader_kwargs) as train_loader, ArgoDataLoader(**val_loader_kwargs) as val_loader:
        train_loop(args, train_loader, val_loader)




def train_loop(args,train_loader,val_loader):
    question_feat_size = 1024
    image_feature_size=2048
    lidar_feature_size=1024

    if args.model_type == 'LSTM_BASIC':
        model = LSTM_BASIC(args, question_feat_size, image_feature_size,lidar_feature_size,num_classes=34, encoder=args.encoder_type)
    elif args.model_type == 'CNN_LSTM':
        model=CNN_LSTM(args,question_feat_size, image_feature_size, num_classes=34, qa=None, encoder=args.encoder_type,method=args.method)
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
    elif args.model_type=='LIDAR_MODEL':
        model=LIDAR_MODEL(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method,grouping=args.grouping)
    elif args.model_type=='MLB':
        question_feat_size=1024
        image_feature_size=512
        model=MLB(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method)
    elif args.model_type=='MUTAN':
        question_feat_size=1024
        image_feature_size=512
        model=MUTAN(args,question_feat_size,image_feature_size,lidar_feature_size,num_classes=34,qa=None,encoder=args.encoder_type,method=args.method)
    else:
        raise NotImplementedError('Given Module is not implemented')
    


    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-3, momentum=0.5)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3, )

    start_epoch = 0
    if args.resume_training:
        data = load_weights(args, model, optimizer)
        if type(data) == list:
            model, optimizer, start_epoch, loss, accuracy = data
            print("Resuming Training: \n Loaded  weights for ")
            print("Epoch: %d, loss: %.3f, Accuracy: %.4f " % (start_epoch, loss, accuracy),flush=True)
        else:
            print(" error occured while loading model training freshly")
            model = data
    
    

    ###########################################################################multiple GPU use#
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    

    model.to(device=args.device)
    # model_name = args.model_type + '_' + args.features_type + '_' + args.encoder_type
    model_name = args.model_type + '_' + args.encoder_type
    
    gap = 1
    running_loss = current_loss = 0
    highest_accuracy = accuracy = 0
    train_iter = val_iter = 0


    for epoch in range(start_epoch+1,args.num_epochs):
        for i, data in enumerate(train_loader):
            question, image_feature,ques_lengths,point_set,answer=data

            question = question.to(device=args.device)
            ques_lengths = ques_lengths.to(device=args.device)
            answer = answer.to(device=args.device)
            image_feature=image_feature.to(device=args.device)
            point_set=point_set.to(device=args.device)


            optimizer.zero_grad()
            pred = model(question, image_feature,ques_lengths,point_set)
            
            loss = criterion(pred, answer)
            loss.backward()
            optimizer.step()


            current_loss = loss.detach().item()
            running_loss += current_loss
            if i % gap == 0 and i!=0:
                print('[%d, %d] loss: %.4f' % (epoch, i + 1, running_loss / gap),flush=True)
                running_loss = 0.0
            train_iter += 1
            gc.collect()



        if args.skip_val:
            print('skipping validation')
            # model_name = args.model_type + '_' + args.features_type + '_' + args.encoder_type
            model_name = args.model_type + '_' + args.encoder_type
            save_model(args, epoch, current_loss, accuracy, model,
                       optimizer, base_name=model_name)
            continue
        
        model.eval()
        total_corrects = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                question, image_feature,ques_lengths,point_set,answer = data

                question = question.to(device=args.device)
                ques_lengths = ques_lengths.to(device=args.device)
                image_feature=image_feature.to(device=args.device)
                point_set=point_set.to(device=args.device)
                
                optimizer.zero_grad()

                pred = model(question, image_feature,ques_lengths,point_set)
                
                predictions = pred.detach().cpu()
                correct_count_vec, correct_count = correct_pred_count(predictions, answer)
                total_corrects += correct_count.item()
                batch_accuracy = correct_count.item() / question.size(0)
                print(str(i), " batch accuracy %.3f", batch_accuracy)
                val_iter += 1
                gc.collect()


        total_samples = val_loader.get_dset_size()
        accuracy = total_corrects / total_samples
        print(" Accuracy: %.3f \n Total samples: %d \n Highest Accuracy: %.3f"
              % (accuracy, total_samples, highest_accuracy),flush=True)

        if highest_accuracy <= accuracy:
            save_model(args, epoch, current_loss, accuracy, model,
                       optimizer, base_name=model_name)
            highest_accuracy = accuracy
        model.train()
        gc.collect()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)







