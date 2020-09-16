import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aqa.module_helper import QuestionModule, AnswerModule ,LidarSsgModule, LidarMsgModule, StackedAttention, MCBAttention, MCBPolling, DANAttention, CoAtt, MutanFusion, MutanFusion2d, MLBFusion
from aqa.futils import get_word2vec, load_vocab
from aqa.embedding import get_embeddings


##########################Simple_LSTM_Based Model#########################################

class LSTM_BASIC(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm'):
        super(LSTM_BASIC, self).__init__()
        self.qa = qa

        # special_words = ["<UNK>"]
        # self.vocab = load_vocab(os.path.join(args.input_base, args.vocab))
        # word_vectors = get_word2vec(os.path.join(args.input_base, args.ques_vectors))

        # padding = vocab['question_token_to_idx']['<NULL>']
        # D = word2vec.vector_size
        # self.embeddings = get_embeddings(self.vocab, word_vectors, special_words)

        self.vocab = load_vocab(os.path.join(args.input_base, args.vocab))
        N = len(self.vocab['question_token_to_idx'])
        D = 200
        padding = self.vocab['question_token_to_idx']['<NULL>']
        self.embeddings = nn.Embedding(N, D, padding_idx=padding)
        self.question_module = QuestionModule(
            ques_feat_size, self.embeddings, encoder)
        self.classifier = AnswerModule(ques_feat_size, num_classes, (256,))

    def forward(self, x, feat, xlen,point_set):  # forward(self,feat,x, xlen):
        del feat
        del point_set
        x = self.question_module(x, q_len=xlen)
        x = self.classifier(x)
        del xlen
        return x



####################################Simple CNN+LSTM Model########################################
class CNN_LSTM(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='dot'):
        super(CNN_LSTM, self).__init__()
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
        self.image_features_resize = nn.Linear(self.image_feat_size, ques_feat_size)
        self.method=method
        if self.method=='dot':
            self.classifier = AnswerModule(ques_feat_size, num_classes,use_batchnorm=True,dropout=0.5)
        if self.method=='concat':
            self.classifier=AnswerModule(ques_feat_size+image_feature_size*7, num_classes,use_batchnorm=True,dropout=0.5)

    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen)
        # print(ques.shape)
        # print(feat.shape)
        # return
        if self.method=='dot':
            joint_feat = feat.prod(dim=1)
            joint_feat=self.image_features_resize(joint_feat)
            x = joint_feat*ques
        if self.method=='concat':
            feat=feat.transpose(0,1)
            joint_feat=torch.cat([feat[i] for i in range(7)],dim=1)
            x=torch.cat([joint_feat,ques],dim=1)

        del feat
        x = self.classifier(x)

        return x
####################################Simple LIDAR Model Model########################################  
class LIDAR_MODEL(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='concat',grouping='single_scale'):
        super(LIDAR_MODEL,self).__init__()
        self.qa = qa
        self.image_feat_size=image_feature_size
        self.vocab = load_vocab(os.path.join(args.input_base, args.vocab))
        N = len(self.vocab['question_token_to_idx'])
        D = 200
        padding = self.vocab['question_token_to_idx']['<NULL>']
        self.embeddings = nn.Embedding(N, D, padding_idx=padding)
        self.question_module = QuestionModule(ques_feat_size, self.embeddings, encoder)
        self.method=method
        self.grouping=grouping
        
        if self.grouping=='single_scale':
            self.lidar_module=LidarSsgModule(normal_channel=False)
        if self.grouping=='multi_scale':
            self.lidar_module=LidarMsgModule(normal_channel=False)


        if self.method=='dot':
            self.classifier = AnswerModule(ques_feat_size, num_classes,use_batchnorm=True,dropout=0.5)
        if self.method=='concat':
            self.classifier=AnswerModule(ques_feat_size+1024, num_classes,use_batchnorm=True,dropout=0.5)

    def forward(self, x, feat, xlen,point_set):
        del feat
        ques = self.question_module(x, q_len=xlen)
        point_set=point_set.float()
        lidar=self.lidar_module(point_set)
        if self.method=='dot':
            x = lidar*ques
        if self.method=='concat':
            x=torch.cat([lidar,ques],dim=1)
        x = self.classifier(x)
        return x

#####################################SAN(StackedAttentionModel)##########################################
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
        self.method=method
        if self.method=='concat':
            self.classifier = AnswerModule(3584, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        if self.method=='hierarchical':
            self.classifier = AnswerModule(512, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        # self.linweights=nn.Linear(512,7)
        self.linweights=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,7)  ,   #no of glimpses
            nn.Sigmoid()
        )
        
    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen)
        att_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14

        for i in range(7):
            img_att=self.attention(ques,feat[i])   #B,512
            att_7.append(img_att)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same

        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)

        if self.method=='hierarchical':
            energies=self.linweights(ques).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)

        x=self.classifier(joint_feat)
        return x


#####################################MCB(MultimodalCompactBilinearPolling)##########################################
class MCB(nn.Module):
    def __init__(self, args, ques_feat_size, image_feature_size, lidar_feature_size,num_classes, qa=None, encoder='lstm',method='concat'):
        super(MCB,self).__init__()
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
        self.attention=MCBAttention(ques_feat_size,512,512,8000)
        self.method=method
        if self.method=='concat':
            self.classifier = AnswerModule(512*8, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        else:
            self.classifier = AnswerModule(8000, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        self.mcb=MCBPolling(512, 8000, n_modalities=2)
        self.linweights=nn.Linear(ques_feat_size    ,7)
        
    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen)
        att_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14

        for i in range(7):
            img_att=self.attention(ques,feat[i])   #B,512
            att_7.append(img_att)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same  #
        #seven soft attention on images
        # print(att_7.shape)
        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)  #B,3584
            joint_feat=torch.cat([joint_feat,ques],dim=1)  #3584+1000

        if self.method=='hierarchical':
            energies=self.linweights(ques).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
            joint_feat=self.mcb(ques,joint_feat)

        x=self.classifier(joint_feat)
        return x


#####################################DAN##########################################

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
        self.linweights=nn.Linear(ques_feat_size ,7)
        self.method=method
        self.softmax=nn.Softmax(dim=1)
        self.tanh=nn.Tanh()
        if self.method=='concat':
            self.classifier = AnswerModule(2*512*7+ques_feat_size, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        else:
            self.classifier = AnswerModule(2*512, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen) #batch,seq_len,1024(2*hidden)
        h0=ques[:,-1,:512]
        ques=ques.transpose(0,1) #seq_len,batch,1024

        att_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14

        for i in range(7):
            img_att=self.attention(ques,feat[i])   #B,512
            att_7.append(img_att)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same  #
        #seven soft attention on images
        # print(h0.shape,att_7.shape)
        # return
        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)  #B,3584
            # print(joint_feat.shape)
            joint_feat=torch.cat([joint_feat,h0],dim=1)  #3584+1000
            # print(joint_feat.shape)

        if self.method=='hierarchical':
            energies=self.softmax(self.tanh(self.linweights(h0))).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
            # joint_feat=joint_feat/torch.norm(joint_feat,p=1).detach()
        x=self.classifier(joint_feat)
        return x

#####################################MFB##########################################

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
        self.method=method
        if self.method=='concat':
            self.classifier = AnswerModule(500*7+ques_feat_size, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3500 if method is concat else 500
        else:
            self.classifier = AnswerModule(500, num_classes,(256,),use_batchnorm=True,dropout=0.5)  #3500 if method is concat else 500
        self.linweights=nn.Linear(ques_feat_size,7)
        


    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen)   #b,seq_len,hidden_dim(*direction*layer)
        h0=ques[:,-1,:512]

        att_7=[]
        feat=feat.transpose(0,1)    #b,7,512,14,14-> 7,B,512,14,14

        for i in range(7):
            img_att=self.attention(ques,feat[i].view(-1,196,512))   #require in form b,h*w,feat_size
            att_7.append(img_att)
        del feat
        att_7=torch.stack(att_7,dim=0)   #b,7,500
        #seven soft attention on images

        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)  #B,3584
            joint_feat=torch.cat([joint_feat,ques])  #3584+1000

        if self.method=='hierarchical':
            energies=self.linweights(h0).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
            # joint_feat=joint_feat/torch.norm(joint_feat,p=1).detach()
        x=self.classifier(joint_feat)
        return x



#####################################MLB and MUTAN##########################################
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

        return list_v_att

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
        list_v_att = self._attention(input_v, x_q_vec)
        x = self._fusion_glimpses(list_v_att, x_q_vec)
        # x = self._classif(x)
        return x


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
        self.method=method
        if self.method=='concat':
            self.classifier = AnswerModule(self.opt['fusion']['dim_h']* self.opt['attention']['nb_glimpses']*7+ques_feat_size, num_classes,(512,256,),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        else:
            self.classifier = AnswerModule(self.opt['fusion']['dim_h']* self.opt['attention']['nb_glimpses'], num_classes,(),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        self.linweights=nn.Linear(ques_feat_size ,7)
        


    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen) 

        att_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14

        for i in range(7):
            img_att=self.attention(ques,feat[i])   #B,512
            att_7.append(img_att)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same  #
        #seven soft attention on images

        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)  #B,3584
            joint_feat=torch.cat([joint_feat,ques],dim=1)  #3584+1000

        if self.method=='hierarchical':
            energies=self.linweights(ques).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
        x=self.classifier(joint_feat)
        return x

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
        self.method=method
        if self.method=='concat':
           self.classifier = AnswerModule(self.opt['fusion']['dim_mm']*7+ques_feat_size, num_classes,(),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        else:
            self.classifier = AnswerModule(self.opt['fusion']['dim_mm'], num_classes,(),use_batchnorm=True,dropout=0.5)  #3584 if method is concat
        self.linweights=nn.Linear(ques_feat_size ,7)
        


    def forward(self, x, feat, xlen,point_set):
        del point_set
        ques = self.question_module(x, q_len=xlen) 

        att_7=[]
        feat=feat.transpose(0,1)  #B,7,512,14,14->7,B,512,14,14

        for i in range(7):
            img_att=self.attention(ques,feat[i])   #B,512
            # print(img_att.shape)
            att_7.append(img_att)
        del feat
        att_7=torch.stack(att_7,dim=0)  #7,B,512-> B,7,512 if dim=1 else same  #
        #seven soft attention on images

        if self.method=='concat':
            joint_feat=torch.cat([att_7[i] for i in range(7)], dim=1)  #B,3584
            joint_feat=torch.cat([joint_feat,ques],dim=1)  #3584+1000

        if self.method=='hierarchical':
            energies=self.linweights(ques).view(-1,1,7)
            joint_feat=torch.bmm(energies,att_7.transpose(0,1)).squeeze(1)
        x=self.classifier(joint_feat)
        return x