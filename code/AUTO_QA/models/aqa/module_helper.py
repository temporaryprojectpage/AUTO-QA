import argparse, os, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



import argparse, json
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


def init_rnn(rnn_type, input_dim, hidden_dim, dropout=0, bidirectional=False):
    if rnn_type == 'gru':
        return nn.GRU(input_dim, hidden_dim, dropout=dropout,
                      batch_first=True, bidirectional=bidirectional)
    elif rnn_type == 'lstm':
        return nn.LSTM(input_dim, hidden_dim, dropout=dropout,
                       batch_first=True, bidirectional=bidirectional)
    else:
        print('RNN type ' + str(rnn_type) + ' not yet implemented.')
        raise (NotImplementedError)


def init_weights(module, init_type='xavier_uniform'):
    if init_type == 'xavier_normal':
        initializer = init.xavier_normal_
    elif init_type == 'xavier_uniform':
        initializer = init.xavier_uniform_
    else:
        print('initializer type not implemented')
        raise NotImplementedError
    for n, p in module.named_parameters():
        if 'bias' in n:
            init.constant_(p, 0.0)
        elif 'weight' in n:
            initializer(p)
    return module


class QuestionModule(nn.Module):
    def __init__(self, hidden_size, embeddings, encoder_type='lstm', bidirectional=False,
                 dropout=0, give_last=True):
        super().__init__()
        self.embedding = embeddings
        self.hidden_dim = hidden_size
        self.encoder_type = encoder_type
        vocab_size, self.embedding_dim = embeddings.weight.shape
        self.encoder_rnn = init_rnn(encoder_type, self.embedding_dim, hidden_size,
                                    dropout, bidirectional)
        self.encoder_rnn = init_weights(self.encoder_rnn, init_type='xavier_uniform')
        self.last = give_last
        self.num_direction=2 if bidirectional else 1
    
    def _init_hidden(self, dim, embed_data):
        N, H = dim
        h0 = torch.zeros(self.num_direction, N, H).type_as(embed_data)
        c0 = None
        if self.encoder_type == 'lstm':
            c0 = torch.zeros(self.num_direction, N, H).type_as(embed_data)
        return (h0, c0)
    
    def get_dims(self, x=None):
        H = self.hidden_dim
        Em = self.embedding_dim
        N = x.size(0) if x is not None else None
        Seq_in = x.size(1) if x is not None else None
        return (N, H, Em, Seq_in)
    
    def forward(self, x, **kwargs):
        N, H, Em, Seq_in = self.get_dims(x)
        x = self.embedding(x)
        h0, c0 = self._init_hidden((N, H), x.data)
        
        # masking the paddings before entering into lstms
        x = rnn_utils.pack_padded_sequence(x, lengths=kwargs['q_len'], batch_first=True)
        self.encoder_rnn.flatten_parameters()
        if self.encoder_type == 'lstm':
            x, (h0, c0) = self.encoder_rnn(x, (h0, c0))
        elif self.encoder_type == 'gru':
            x, h0 = self.encoder_rnn(x, h0)
        else:
            print('encoder type not implemented')
            raise NotImplementedError
        
        # pad_packed_sequence return unpacked sequence and length of each sequence
        x, lengths = rnn_utils.pad_packed_sequence(x, batch_first=True, total_length=Seq_in)
        
        if self.last:
            x = h0 if self.encoder_type == 'gru' else c0
            x = x.transpose(0, 1).squeeze(1)
        return x




# class AnswerModule(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=(256,), use_softmax=False):
#         super().__init__()
#         self.use_softmax = use_softmax
#         self.layer = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim[0]),
#             nn.Linear(hidden_dim, hidden_dim[0]),
#             nn.Linear(hidden_dim, output_dim[0])
#         )
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         x = self.layer(x)
#         if self.use_softmax:
#             x = self.softmax(x)
#         return x


class AnswerModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(256,), use_batchnorm=False,dropout=0,use_softmax=False):
        super().__init__()
        self.use_softmax = use_softmax
        layers=[]
        D = input_dim
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(input_dim))
        for dim in hidden_dim:
            layers.append(nn.Linear(D, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))
            D = dim
        layers.append(nn.Linear(D, output_dim))

        self.layer=nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.layer(x)
        if self.use_softmax:
            x = self.softmax(x)
        return x





from aqa.pointnet_util import PointNetSetAbstraction,PointNetSetAbstractionMsg


class LidarSsgModule(nn.Module):
    def __init__(self,normal_channel=False):
        super(LidarSsgModule,self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        return x

class LidarMsgModule(nn.Module):
    def __init__(self,normal_channel=False):
        super(LidarMsgModule,self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        return x

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
        return img_att

    def forward(self,x,feat):
        #stacking attention layer as per paper
        for i in range(self.num_stacked_attention):
            u=self.loop_attention(x,feat)
            x=u+x
        return x

class MCBPolling(nn.Module):
    """
    Multimodal Compact Bilinear Pooling Module
    """

    def __init__(self, original_dim, projection_dim, n_modalities=2):
        super(MCBPolling, self).__init__()
        
        self.C = []
        self.n_modalities = n_modalities
        
        for _ in range(n_modalities):
            # C tensor performs the mapping of the h vector and stores the s vector values as well
            C = torch.zeros(original_dim, projection_dim)
            for i in range(original_dim):
                C[i, np.random.randint(0, projection_dim - 1)] = 2 * np.random.randint(0, 2) - 1  # s values
                
                if torch.cuda.is_available():
                    C = C.cuda()
            
            self.C.append(C)
    
    def forward(self, *x):
        feature_size = x[0].size()
        y = [0] * self.n_modalities
        
        for i, d in enumerate(x):
            y[i] = d.mm(self.C[i]).view(feature_size[0], -1)
        
        phi = y[0]
        signal_sizes = y[0].size()[1:]  # signal_sizes should not have batch dimension as per docs
        
        for i in range(1, self.n_modalities):
            i_fft = torch.rfft(phi, 1)
            j_fft = torch.rfft(y[i], 1)
            
            # element wise multiplication
            x = i_fft.mul(j_fft)
            
            # inverse FFT
            phi = torch.irfft(x, 1, signal_sizes=signal_sizes)
        
        return phi


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

        # K indicates the number of hops
        for k in range(self.k):
            # Compute Visual Attention
            hv = self.tanh(self.Wv(self.dropout(vns))) * self.tanh(self.Wvm(self.dropout(memory)))
            # attention weights for every region
            alphaV = self.softmax(self.Wvh(self.dropout(hv))) #(seq_len, batch_size, memory_size)
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
        return memory
####################################################MFB#######################################################
class MFB(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
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
        self.mfb = MFB(img_feat_size, ques_att_feat_size, True)
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

        return iatt_feat


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
            self.mfh1 = MFB(img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(img_att_feat_size, ques_att_feat_size, False)
        else:  # MFB
            self.mfb = MFB(img_att_feat_size, ques_att_feat_size, True)

    def forward(self, ques_feat,img_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)
        # return fuse_feat
        if self.high_order:  # MFH
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # (N, 2*O)
        else:  # MFB
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
            z = z.squeeze(1)                                                            # (N, O)

        return z


########################################MLB and MUTAN###############################################
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
