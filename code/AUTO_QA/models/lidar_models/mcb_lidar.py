import copy
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aqa.module_helper import QuestionModule, AnswerModule,LidarSsgModule, LidarMsgModule
from aqa.futils import get_word2vec, load_vocab
from aqa.embedding import get_embeddings

class MCBAttention(nn.Module):
    def __init__(self, ques_feat_size,feat_size,mid_features,mcb_dim):
        super(MCBAttention, self).__init__()
        self.mid_features = mid_features
        self.mcb_dim = mcb_dim
        self.mcb = MCBPolling(self.mid_features, mcb_dim, n_modalities=2)
        self.attention = nn.Sequential(
            nn.Conv2d(mcb_dim, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.conv1 = nn.Conv2d(feat_size, self.mid_features, 1, bias=False)  
        self.lin1 = nn.Linear(ques_feat_size, self.mid_features)
    def forward(self, x, feat):
        B=x.size()[0]
        H,W=feat.size()[-2:]

        v=self.conv1(nn.Dropout2d(p=0.5)(feat))
        v=v.view(-1,H*W,self.mid_features)
        v_norm=v/torch.norm(v,p=2).detach()
        v_norm=v_norm.view(-1,H,W,self.mid_features)

        q=self.lin1(nn.Dropout(p=0.5)(x))
        q=q.unsqueeze(1)
        q_tiled=q.repeat(1,H*W,1)  
        q_tiled=q_tiled.view(-1,H,W,self.mid_features)

        q_tiled_mcb=q_tiled.contiguous().view(-1,self.mid_features)
        v_norm_mcb=v_norm.contiguous().view(-1,self.mid_features)

        att_x=self.mcb(q_tiled_mcb,v_norm_mcb)
        att_x=att_x.view(-1,H,W,self.mcb_dim)
        att_x=att_x.permute(0,3,1,2)

        att_x = self.attention(att_x)
        
        att_x = att_x.repeat(1, self.mid_features, 1, 1)  # BxDxHxW
        # print('img_feat_norm',img_feat_norm.shape)# img_feat_norm torch.Size([120, 14, 14, 512])
        v_norm = v_norm.permute(0, 3, 1, 2).contiguous()  # BxDxHxW
        # print('img_feat_norm',img_feat_norm.shape)# img_feat_norm torch.Size([120, 512, 14, 14])
        img_att_x = v_norm.mul(att_x)
        # print('img_att_x',img_att_x.shape)img_att_x torch.Size([120, 512, 14, 14])
        img_att_x = img_att_x.sum(dim=3).sum(dim=2)
        # img_att=torch.bmm(energies.view(-1,1,H*W),v_norm.view(-1,H*W,self.mid_features))
        return img_att_x





class MCB_LIDAR(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',grouping='single_scale'):
        super(MCB_LIDAR,self).__init__()
        self.qa = qa
        self.image_feat_size=image_feature_size
        self.vocab = load_vocab(os.path.join(args.input_base, args.vocab))
        N = len(self.vocab['question_token_to_idx'])
        D = 200
        padding = self.vocab['question_token_to_idx']['<NULL>']
        self.embeddings = nn.Embedding(N, D, padding_idx=padding)
        self.question_module = QuestionModule(ques_feat_size, self.embeddings, encoder)
        self.attention=MCBAttention(ques_feat_size,512,512,8000)
        self.classifier = AnswerModule(8000+ques_feat_size+1024, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        self.mcb=MCBPolling(512, 8000, n_modalities=2)
        self.linweights=nn.Linear(ques_feat_size    ,7)
        self.grouping=grouping
        if self.grouping=='single_scale':
            self.lidar_module=LidarSsgModule(normal_channel=False)
        if self.grouping=='single_scale':
            self.lidar_module=LidarMsgModule(normal_channel=False)
    def forward(self, x, feat, xlen,point_set):

        ques = self.question_module(x, q_len=xlen)
        att_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14
        wgt_7=[]
        for i in range(7):
            img_att=self.attention(ques,feat[i])   #B,512
            att_7.append(img_att)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same  #
        #seven soft attention on images
        energies=self.linweights(ques).view(-1,1,7)
        joint_feat1=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
        joint_feat1=self.mcb(ques,joint_feat1)




         #lidar pipeline
        point_set=point_set.float()
        lidar=self.lidar_module(point_set)
        joint_feat2=torch.cat([lidar,ques],dim=1)


        #classifier
        joint_feat=torch.cat([joint_feat1,joint_feat2],dim=1)  #1024+ques_feat_size=512
        x=self.classifier(joint_feat)
        return x

