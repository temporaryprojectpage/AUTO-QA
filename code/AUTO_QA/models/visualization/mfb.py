import copy
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aqa.module_helper import QuestionModule, AnswerModule 
from aqa.futils import get_word2vec, load_vocab
from aqa.embedding import get_embeddings


class MFBPolling(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size, is_first):
        super(MFBPolling, self).__init__()
        self.K=5
        self.O=500
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, self.K * self.O)  #k*0
        self.proj_q = nn.Linear(ques_feat_size, self.K * self.O)
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AvgPool1d(self.K ,stride=self.K)  #k
    
    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O    )
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat                  # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.O)      # (N, C, O)
        return z, exp_out

class QAtt(nn.Module):
    def __init__(self,ques_feat_size,n_glimpses):
        super(QAtt, self).__init__()
        # self.mlp = MLP(
        #     in_size=__C.LSTM_OUT_SIZE,
        #     mid_size=__C.HIDDEN_SIZE,
        #     out_size=__C.Q_GLIMPSES,
        #     dropout_r=__C.DROPOUT_R,
        #     use_relu=True
        # )
        self.mlp=nn.Sequential(
            nn.Linear(ques_feat_size,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,n_glimpses)     #no of glimpses
        )
        self.n_glimpses=n_glimpses
    def forward(self, ques_feat):
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        '''
        qatt_maps = self.mlp(ques_feat)                 # (N, T, Q_GLIMPSES)
        qatt_maps = F.softmax(qatt_maps, dim=1)         # (N, T, Q_GLIMPSES)

        qatt_feat_list = []
        for i in range(self.n_glimpses):
            mask = qatt_maps[:, :, i:i + 1]             # (N, T, 1)
            mask = mask * ques_feat                     # (N, T, LSTM_OUT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, LSTM_OUT_SIZE)
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)    # (N, LSTM_OUT_SIZE*Q_GLIMPSES)

        return qatt_feat    
class IAtt(nn.Module):
    def __init__(self,img_feat_size, ques_att_feat_size,n_glimpses):
        super(IAtt, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.mfb = MFBPolling(img_feat_size, ques_att_feat_size, True)
        self.mlp=nn.Sequential(
            nn.Linear(500,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512,n_glimpses)     
        )
        # self.mlp = MLP(
        #     in_size=__C.MFB_O,
        #     mid_size=__C.HIDDEN_SIZE,
        #     out_size=__C.I_GLIMPSES,
        #     dropout_r=__C.DROPOUT_R,
        #     use_relu=True
        # )
        self.n_glimpses=n_glimpses
    def forward(self, img_feat, ques_att_feat):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)      # (N, 1, LSTM_OUT_SIZE * Q_GLIMPSES)
        img_feat = self.dropout(img_feat)
        z, _ = self.mfb(img_feat, ques_att_feat)        # (N, C, O)

        iatt_maps = self.mlp(z)                         # (N, C, I_GLIMPSES)
        iatt_maps = F.softmax(iatt_maps, dim=1)         # (N, C, I_GLIMPSES)

        iatt_feat_list = []
        for i in range(self.n_glimpses):
            mask = iatt_maps[:, :, i:i + 1]             # (N, C, 1)
            mask = mask * img_feat                      # (N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        return iatt_feat,iatt_maps


class CoAtt(nn.Module):
    def __init__(self,ques_feat_size,feat_size,n_glimpses,high_order=False):
        super(CoAtt, self).__init__()
        
        img_feat_size = feat_size
        img_att_feat_size = img_feat_size * n_glimpses
        ques_att_feat_size = ques_feat_size * n_glimpses

        self.q_att = QAtt(ques_feat_size,n_glimpses)
        self.i_att = IAtt(img_feat_size, ques_att_feat_size,n_glimpses)
        self.high_order=high_order
        if self.high_order:  # MFH
            self.mfh1 = MFBPolling(img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFBPolling(img_att_feat_size, ques_att_feat_size, False)
        else:  # MFBPolling
            self.mfb = MFBPolling(img_att_feat_size, ques_att_feat_size, True)

    def forward(self, ques_feat,img_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat,iatt_maps = self.i_att(img_feat, ques_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)
        # return fuse_feat
        if self.high_order:  # MFH
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # (N, 2*O)
        else:  # MFB
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
            z = z.squeeze(1)                                                            # (N, O)

        return z,iatt_maps


class MFB(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='concat'):
        super(MFB,self).__init__()
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
        self.question_module = QuestionModule(ques_feat_size, self.embeddings, encoder,give_last=False)
        self.attention=CoAtt(ques_feat_size,image_feature_size,2)
        self.classifier = AnswerModule(500, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3500 if method is concat else 500
        self.linweights=nn.Linear(ques_feat_size,7)
        self.method=method


    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen)   #b,seq_len,hidden_dim(*direction*layer)
        h0=ques[:,-1,:512]

        att_7=[]
        wgt_7=[]
        feat=feat.transpose(0,1)    #b,7,512,14,14-> 7,B,512,14,14

        for i in range(7):
            img_att,wgt=self.attention(ques,feat[i].view(-1,196,512))   #require in form b,h*w,feat_size
            att_7.append(img_att)
            wgt_7.append(wgt)
        del feat
        att_7=torch.stack(att_7,dim=0)   #b,7,500
        #seven soft attention on images
        wgt_7=torch.stack(wgt_7,dim=1)
        # print(wgt_7.shape)  ##b,7,196,2
        # return

        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)  #B,3584
            # joint_feat=torch.cat([joint_feat,ques])  #3584+1000

        if self.method=='hierarchical':
            energies=self.linweights(h0).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
            # joint_feat=joint_feat/torch.norm(joint_feat,p=1).detach()
        x=self.classifier(joint_feat)
        return x,wgt_7,energies
