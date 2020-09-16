import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aqa.module_helper import QuestionModule, AnswerModule 
from aqa.futils import get_word2vec, load_vocab
from aqa.embedding import get_embeddings


class DANAttention(nn.Module):
    def __init__(self,ques_feat_size,feat_size,hidden_size,n_hops):
        super(DANAttention,self).__init__()
        memory_size=2*hidden_size

        #visual attention
        self.Wv = nn.Linear(in_features=feat_size, out_features=hidden_size)
        self.Wvm = nn.Linear(in_features=memory_size, out_features=hidden_size)
        self.Wvh = nn.Linear(in_features=hidden_size, out_features=1)
        self.P = nn.Linear(in_features=feat_size, out_features=memory_size)
    
        # Textual Attention
        self.Wu = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        self.Wum = nn.Linear(in_features=memory_size, out_features=hidden_size)
        self.Wuh = nn.Linear(in_features=hidden_size, out_features=1)
 
        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Activations
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0) # Softmax over first dimension

        # Loops
        self.k = n_hops
    def forward(self,x,feat):
        B=feat.size(0)
        feat_dim=feat.size(1)#feat 512,14,14
        feat = feat.view(B,feat_dim, -1)
        vns = feat.permute(2,0,1) #(nregion,B,dim)
        uts=x
        # print(vns.shape)
        u = x.mean(0)
        v = self.tanh( self.P( vns.mean(0) ))
        # print(v.shape)
        memory = v * u
        wgt=[]
        # K indicates the number of hops
        for k in range(self.k):
            # Compute Visual Attention
            hv = self.tanh(self.Wv(self.dropout(vns))) * self.tanh(self.Wvm(self.dropout(memory)))
            # attention weights for every region
            alphaV = self.softmax(self.Wvh(self.dropout(hv))) #(seq_len, batch_size, memory_size)
            wgt.append(alphaV.view(B,-1))
            # Sum over regions
            v = self.tanh(self.P(alphaV * vns)).sum(0)

            # Text 
            # (seq_len, batch_size, dim) * (batch_size, dim)
            hu = self.tanh(self.Wu(self.dropout(uts))) * self.tanh(self.Wum(self.dropout(memory)))
            # attention weights for text features
            alphaU = self.softmax(self.Wuh(self.dropout(hu)))  # (seq_len, batch_size, memory_size)
            # Sum over sequence
            u = (alphaU * uts).sum(0) # Sum over sequence
            
            # Build Memory
            memory = memory + u * v
        wgt=torch.stack(wgt,dim=1)
        return memory,wgt

class DAN(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='concat'):
        super(DAN,self).__init__()
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
        self.question_module = QuestionModule(ques_feat_size, self.embeddings, encoder,bidirectional=True,give_last=False)
        self.attention=DANAttention(ques_feat_size,512,512,2)
        self.classifier = AnswerModule(2*512, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        self.linweights=nn.Linear(ques_feat_size ,7)
        self.method=method
        self.softmax=nn.Softmax(dim=1)
        self.tanh=nn.Tanh()


    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen) #batch,seq_len,1024(2*hidden)
        h0=ques[:,-1,:512]
        ques=ques.transpose(0,1) #seq_len,batch,1024

        att_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14
        wgt_7=[]
        for i in range(7):
            img_att,wgt=self.attention(ques,feat[i])   #B,512
            att_7.append(img_att)
            wgt_7.append(wgt)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same  #
        #seven soft attention on images
        wgt_7=torch.stack(wgt_7,dim=1)
        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)  #B,3584
            joint_feat=torch.cat([join_feat,ques])  #3584+1000

        if self.method=='hierarchical':
            energies=self.softmax(self.tanh(self.linweights(h0))).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
            # joint_feat=joint_feat/torch.norm(joint_feat,p=1).detach()
        x=self.classifier(joint_feat)
        return x,wgt_7,energies

# class SAN(nn.Module):
#     def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='concat'):
#         super(SAN,self).__init__()
#         self.qa = qa

#         # special_words = ["<UNK>"]
#         # self.vocab = load_vocab(os.path.join(args.input_base, args.vocab))
#         # word_vectors = get_word2vec(os.path.join(args.input_base, args.ques_vectors))

#         # padding = vocab['question_token_to_idx']['<NULL>']
#         # D = word2vec.vector_size
#         # self.embeddings = get_embeddings(self.vocab, word_vectors, special_words)
#         self.image_feat_size=image_feature_size
#         self.vocab = load_vocab(os.path.join(args.input_base, args.vocab))
#         N = len(self.vocab['question_token_to_idx'])
#         D = 200
#         padding = self.vocab['question_token_to_idx']['<NULL>']
#         self.embeddings = nn.Embedding(N, D, padding_idx=padding)
#         self.question_module = QuestionModule(ques_feat_size, self.embeddings, encoder)
#         self.attention=StackedAttention(ques_feat_size,512,512,2)
#         self.classifier = AnswerModule(512, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
#         # self.linweights=nn.Linear(512,7)
#         self.linweights=nn.Sequential(
#             nn.Linear(512,256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256,7)  ,   #no of glimpses
#             nn.Sigmoid()
#         )
#         self.method=method
#     def forward(self, x, feat, xlen,point_set):
#         del point_set
#         ques = self.question_module(x, q_len=xlen)
#         att_7=[]
#         feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14
#         wgt_7=[]
#         for i in range(7):
#             img_att,wgt=self.attention(ques,feat[i])   #B,512
#             att_7.append(img_att)
#             wgt_7.append(wgt)
#         del feat
#         att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same
#         wgt_7=torch.stack(wgt_7,dim=1)

#         if self.method=='concat':
#             joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)

#         if self.method=='hierarchical':
#             energies=self.linweights(ques).view(-1,1,7)
#             joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)

#         x=self.classifier(joint_feat)
#         return x,wgt_7,energies

