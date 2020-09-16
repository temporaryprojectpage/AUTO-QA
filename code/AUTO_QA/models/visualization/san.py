import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aqa.module_helper import QuestionModule, AnswerModule 
from aqa.futils import get_word2vec, load_vocab
from aqa.embedding import get_embeddings


class StackedAttention(nn.Module):
    def __init__(self,ques_feat_size,feat_size,mid_features,num_stacked_attention):
        super(StackedAttention,self).__init__()
        self.conv1=nn.Conv2d(feat_size,mid_features,kernel_size=1,padding=0,bias=False)
        self.lin1=nn.Linear(ques_feat_size,mid_features)
        self.conv2=nn.Conv2d(mid_features,1,kernel_size=1,padding=0)
        self.mid_features=mid_features
        self.num_stacked_attention=num_stacked_attention

    def loop_attention(self,x,feat):
        B=x.size()[0]
        H,W=feat.size()[-2:]

        
        v=self.conv1(feat)
        q=self.lin1(x)

        

        q=q.view(B,self.mid_features,1,1).expand(B,self.mid_features,H,W)
        h=torch.tanh(q+v)

        h=self.conv2(h)
        h=h.view(-1,196)
        p=F.softmax(h,dim=1)

        img_att=torch.bmm(p.view(-1,1,196),feat.view(-1,196,self.mid_features))
        img_att=img_att.view(-1,self.mid_features)
        return img_att,p

    def forward(self,x,feat):
        #stacking attention layer as per paper
        wgt=[]
        for i in range(self.num_stacked_attention):
            u,p=self.loop_attention(x,feat)
            x=u+x
            wgt.append(p)
#             print(len(wgt))
        wgt=torch.stack(wgt,dim=1)
#         print(wgt.shape)
        return x,wgt


class SAN(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='concat'):
        super(SAN,self).__init__()
        self.qa = qa

        # special_words = ["<UNK>"]
        # self.vocab = load_vocab(os.path.join(args.input_base, args.vocab))
        # word_vectors = get_word2vec(os.path.join(args.input_base, args.ques_vectors))

        # padding = vocab['question_token_to_idx']['<NULL>']
        # D = word2vec.vector_size
        # self.embeddings = get_embeddings(self.vocab, word_vectors, special_words)
        self.image_feat_size=image_feature_size
        self.vocab = load_vocab(os.path.join(args.input_base, args.vocab))
        N = len(self.vocab['question_token_to_idx'])
        D = 200
        padding = self.vocab['question_token_to_idx']['<NULL>']
        self.embeddings = nn.Embedding(N, D, padding_idx=padding)
        self.question_module = QuestionModule(ques_feat_size, self.embeddings, encoder)
        self.attention=StackedAttention(ques_feat_size,512,512,2)
        self.classifier = AnswerModule(512, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        # self.linweights=nn.Linear(512,7)
        self.linweights=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,7)  ,   #no of glimpses
            nn.Sigmoid()
        )
        self.method=method
    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen)
        att_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14
        wgt_7=[]
        for i in range(7):
            img_att,wgt=self.attention(ques,feat[i])   #B,512
            att_7.append(img_att)
            wgt_7.append(wgt)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same
        wgt_7=torch.stack(wgt_7,dim=1)

        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)

        if self.method=='hierarchical':
            energies=self.linweights(ques).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)

        x=self.classifier(joint_feat)
        return x,wgt_7,energies

