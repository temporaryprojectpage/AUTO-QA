import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError


class MLBFusion(AbstractFusion):

    def __init__(self, opt):
        super(MLBFusion, self).__init__(opt)
        # Modules
        if 'dim_v' in self.opt:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_h'])
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if 'dim_q' in self.opt:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_h'])
        else:
            print('Warning fusion.py: no question embedding before fusion')
        
    def forward(self, input_v, input_q):
        # visual (cnn features)
        if 'dim_v' in self.opt:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v
        # question (rnn features)
        if 'dim_q' in self.opt:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q
        # hadamard product
        x_mm = torch.mul(x_q, x_v)
        return x_mm


class MutanFusion(AbstractFusion):

    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__(opt)
        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        # Modules
        if self.visual_embedding:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_hv'])
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if self.question_embedding:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_hq'])
        else:
            print('Warning fusion.py: no question embedding before fusion')
        
        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.opt['dim_hv'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(self.opt['dim_hq'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError
        batch_size = input_v.size(0)

        if self.visual_embedding:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                    x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v

        if self.question_embedding:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                    x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q

        x_mm = []
        for i in range(self.opt['R']):

            x_hv = F.dropout(x_v, p=self.opt['dropout_hv'], training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            if 'activation_hv' in self.opt:
                x_hv = getattr(F, self.opt['activation_hv'])(x_hv)

            x_hq = F.dropout(x_q, p=self.opt['dropout_hq'], training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            if 'activation_hq' in self.opt:
                x_hq = getattr(F, self.opt['activation_hq'])(x_hq)

            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.opt['dim_mm'])

        if 'activation_mm' in self.opt:
            x_mm = getattr(F, self.opt['activation_mm'])(x_mm)

        return x_mm


class MutanFusion2d(MutanFusion):

    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion2d, self).__init__(opt,
                                            visual_embedding,
                                            question_embedding)

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 3:
            raise ValueError
        batch_size = input_v.size(0)
        weight_height = input_v.size(1)
        dim_hv = input_v.size(2)
        dim_hq = input_q.size(2)
        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_q.is_contiguous():
            input_q = input_q.contiguous()
        x_v = input_v.view(batch_size * weight_height, self.opt['dim_hv'])
        x_q = input_q.view(batch_size * weight_height, self.opt['dim_hq'])
        x_mm = super().forward(x_v, x_q)
        x_mm = x_mm.view(batch_size, weight_height, self.opt['dim_mm'])
        return x_mm


class AbstractAtt(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(AbstractAtt, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        # Modules
        # self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])
        # Modules for attention
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'],
                                    self.opt['attention']['dim_v'], 1, 1)
        self.linear_q_att = nn.Linear(self.opt['dim_q'],
                                      self.opt['attention']['dim_q'])
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_mm'],
                                  self.opt['attention']['nb_glimpses'], 1, 1)
        # Modules for classification
        self.list_linear_v_fusion = None
        self.linear_q_fusion = None
        self.linear_classif = None

    def _fusion_att(self, x_v, x_q):
        raise NotImplementedError

    def _fusion_classif(self, x_v, x_q):
        raise NotImplementedError

    def _attention(self, input_v, x_q_vec):
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)

        # Process visual before fusion
        #x_v = input_v.view(batch_size*width*height, dim_features)
        x_v = input_v
        x_v = F.dropout(x_v,
                        p=self.opt['attention']['dropout_v'],
                        training=self.training)
        x_v = self.conv_v_att(x_v)
        if 'activation_v' in self.opt['attention']:
            x_v = getattr(F, self.opt['attention']['activation_v'])(x_v)
        x_v = x_v.view(batch_size,
                       self.opt['attention']['dim_v'],
                       width * height)
        x_v = x_v.transpose(1,2)

        # Process question before fusion
        x_q = F.dropout(x_q_vec, p=self.opt['attention']['dropout_q'],
                           training=self.training)
        x_q = self.linear_q_att(x_q)
        if 'activation_q' in self.opt['attention']:
            x_q = getattr(F, self.opt['attention']['activation_q'])(x_q)
        x_q = x_q.view(batch_size,
                       1,
                       self.opt['attention']['dim_q'])
        x_q = x_q.expand(batch_size,
                         width * height,
                         self.opt['attention']['dim_q'])

        # First multimodal fusion
        x_att = self._fusion_att(x_v, x_q)

        if 'activation_mm' in self.opt['attention']:
            x_att = getattr(F, self.opt['attention']['activation_mm'])(x_att)

        # Process attention vectors
        x_att = F.dropout(x_att,
                          p=self.opt['attention']['dropout_mm'],
                          training=self.training)
        # can be optim to avoid two views and transposes
        x_att = x_att.view(batch_size,
                           width,
                           height,
                           self.opt['attention']['dim_mm']) 
        x_att = x_att.transpose(2,3).transpose(1,2)
        x_att = self.conv_att(x_att)
        x_att = x_att.view(batch_size,
                           self.opt['attention']['nb_glimpses'],
                           width * height)
        wgt=copy.deepcopy(x_att)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width*height)
            x_att = F.softmax(x_att)
            list_att.append(x_att)

        self.list_att = [x_att.data for x_att in list_att]
        

        # Apply attention vectors to input_v
        x_v = input_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v = x_v.transpose(1,2)

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size,
                               width * height,
                               1)
            x_att = x_att.expand(batch_size,
                                 width * height,
                                 self.opt['dim_v'])
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(batch_size, self.opt['dim_v'])
            list_v_att.append(x_v_att)

        return list_v_att,wgt

    def _fusion_glimpses(self, list_v_att, x_q_vec):
        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att,
                            p=self.opt['fusion']['dropout_v'],
                            training=self.training)
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            if 'activation_v' in self.opt['fusion']:
                x_v = getattr(F, self.opt['fusion']['activation_v'])(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        # Process question
        x_q = F.dropout(x_q_vec,
                        p=self.opt['fusion']['dropout_q'],
                        training=self.training)
        x_q = self.linear_q_fusion(x_q)
        if 'activation_q' in self.opt['fusion']:
            x_q = getattr(F, self.opt['fusion']['activation_q'])(x_q)

        # Second multimodal fusion
        x = self._fusion_classif(x_v, x_q)
        return x

    # def _classif(self, x):

    #     if 'activation' in self.opt['classif']:
    #         x = getattr(F, self.opt['classif']['activation'])(x)
    #     x = F.dropout(x,
    #                   p=self.opt['classif']['dropout'],
    #                   training=self.training)
    #     x = self.linear_classif(x)
    #     return x

    def forward(self, input_q,input_v):
        if input_v.dim() != 4 and input_q.dim() != 2:
            raise ValueError

        x_q_vec = input_q
        list_v_att,wgt = self._attention(input_v, x_q_vec)
        x = self._fusion_glimpses(list_v_att, x_q_vec)
        # x = self._classif(x)
        return x,wgt


class MLBAtt(AbstractAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        # TODO: deep copy ?
        opt['attention']['dim_v']  = opt['attention']['dim_h']
        opt['attention']['dim_q']  = opt['attention']['dim_h']
        opt['attention']['dim_mm'] = opt['attention']['dim_h']
        super(MLBAtt, self).__init__(opt, vocab_words, vocab_answers)
        # Modules for classification
        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(self.opt['dim_v'],
                      self.opt['fusion']['dim_h'])
            for i in range(self.opt['attention']['nb_glimpses'])])
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                         self.opt['fusion']['dim_h']
                                         * self.opt['attention']['nb_glimpses'])
        # self.linear_classif = nn.Linear(self.opt['fusion']['dim_h']
                                        # * self.opt['attention']['nb_glimpses'],
                                        # self.num_classes)

    def _fusion_att(self, x_v, x_q):
        x_att = torch.mul(x_v, x_q)
        return x_att

    def _fusion_classif(self, x_v, x_q):
        x_mm = torch.mul(x_v, x_q)
        return x_mm


class MutanAtt(AbstractAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        # TODO: deep copy ?
        opt['attention']['dim_v'] = opt['attention']['dim_hv']
        opt['attention']['dim_q'] = opt['attention']['dim_hq']
        super(MutanAtt, self).__init__(opt, vocab_words, vocab_answers)
        # Modules for classification
        self.fusion_att = MutanFusion2d(self.opt['attention'],
                                               visual_embedding=False,
                                               question_embedding=False)
        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(self.opt['dim_v'],
                      int(self.opt['fusion']['dim_hv']
                          / opt['attention']['nb_glimpses']))
            for i in range(self.opt['attention']['nb_glimpses'])])
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                         self.opt['fusion']['dim_hq'])
        # self.linear_classif = nn.Linear(self.opt['fusion']['dim_mm'],
        #                                 self.num_classes)
        self.fusion_classif = MutanFusion(self.opt['fusion'],
                                                 visual_embedding=False,
                                                 question_embedding=False)

    def _fusion_att(self, x_v, x_q):
        return self.fusion_att(x_v, x_q)

    def _fusion_classif(self, x_v, x_q):
        return self.fusion_classif(x_v, x_q)


#########################################################################################################################
class MLB(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='concat'):
        super(MLB,self).__init__()
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
        self.opt=dict(
            dim_v=512,
            dim_q = 1024,
            attention=dict(
                nb_glimpses= 4,
                dim_h= 1200,
                dropout_v= 0.5,
                dropout_q= 0.5,
                dropout_mm= 0.5,
                activation_v= "tanh",
                activation_q= "tanh",
                activation_mm= "tanh"),
            fusion=dict(
                dim_h= 1200,
                dropout_v= 0.5,
                dropout_q= 0.5,
                activation_v= "tanh",
                activation_q= "tanh")

        )
        self.attention=MLBAtt(self.opt)
        self.classifier = AnswerModule(self.opt['fusion']['dim_h']* self.opt['attention']['nb_glimpses'], num_classes,(),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        self.linweights=nn.Linear(ques_feat_size ,7)
        self.method=method


    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen) 

        att_7=[]
        wgt_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14

        for i in range(7):
            img_att,wgt=self.attention(ques,feat[i])   #B,512
            att_7.append(img_att)
            wgt_7.append(wgt)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same  #
        #seven soft attention on images
        wgt_7=torch.stack(wgt_7,dim=1)#[100, 7, 4, 196]
        # print(wgt_7.shape)
        # return

        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)  #B,3584
            joint_feat=torch.cat([join_feat,ques])  #3584+1000

        if self.method=='hierarchical':
            energies=self.linweights(ques).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
        x=self.classifier(joint_feat)
        return x,wgt_7,energies

class MUTAN(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='concat'):
        super(MUTAN,self).__init__()
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
        self.opt=dict(
            dim_v= 512,
            dim_q= 1024,
            attention=dict(
                nb_glimpses= 2,
                dim_hv= 310,
                dim_hq= 310,
                dim_mm= 510,
                R= 5,
                dropout_v= 0.5,
                dropout_q= 0.5,
                dropout_mm= 0.5,
                activation_v= "tanh",
                activation_q= "tanh",
                dropout_hv= 0,
                dropout_hq= 0),
            fusion=dict(
                dim_hv= 620,
                dim_hq= 310,
                dim_mm= 510,
                R= 5,
                dropout_v= 0.5,
                dropout_q= 0.5,
                activation_v= "tanh",
                activation_q= "tanh",
                dropout_hv= 0,
                dropout_hq= 0)
        )
        self.attention=MutanAtt(self.opt)
        self.classifier = AnswerModule(self.opt['fusion']['dim_mm'], num_classes,(),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        self.linweights=nn.Linear(ques_feat_size ,7)
        self.method=method


    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen) 

        att_7=[]
        wgt_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14

        for i in range(7):
            img_att,wgt=self.attention(ques,feat[i])   #B,512
            # print(img_att.shape)
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
            energies=self.linweights(ques).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
        x=self.classifier(joint_feat)
        return x,wgt_7,energies