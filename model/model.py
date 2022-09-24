import torch.nn as nn
import torch.nn.functional as F
import torch
from facenet_pytorch import InceptionResnetV1
from base import BaseModel
import torchvision.models as models

###################   CNN + LSTM   ####################################

class TimeWarp(nn.Module):
    def __init__(self, baseModel, method='sqeeze'):
        super(TimeWarp, self).__init__()
        self.baseModel = baseModel
        self.method = method
 
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        if self.method == 'loop':
            output = []
            for i in range(time_steps):
                #input one frame at a time into the basemodel
                x_t = self.baseModel(x[:, i, :, :, :])
                # Flatten the output
                x_t = x_t.view(x_t.size(0), -1)
                output.append(x_t)
            #end loop
            #make output as  ( samples, timesteps, output_size)
            x = torch.stack(output, dim=0).transpose_(0, 1)
            output = None # clear var to reduce data  in memory
            x_t = None  # clear var to reduce data  in memory
        else:
            # reshape input  to be (batch_size * timesteps, input_size)
            x = x.contiguous().view(batch_size * time_steps, C, H, W)
            x = self.baseModel(x)
            x = x.view(x.size(0), -1)
            #make output as  ( samples, timesteps, output_size)
            x = x.contiguous().view(batch_size , time_steps , x.size(-1))
        return x

class extractlastcell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]



#########################################################
#                                                       #
# Authors: Jean Kossaifi, Antoine Toisoul, Adrian Bulat #
#                                                       #
#########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

nn.InstanceNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.InstanceNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.InstanceNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.InstanceNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest') 

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class EmoNet(nn.Module):
    def __init__(self, num_modules=2, n_expression=8, n_reg=2, n_blocks=4, attention=True, temporal_smoothing=False):
        super(EmoNet, self).__init__()
        self.num_modules = num_modules
        self.n_expression = n_expression
        self.n_reg = n_reg
        self.attention = attention
        self.temporal_smoothing = temporal_smoothing
        self.init_smoothing = False

        if self.temporal_smoothing:
            self.n_temporal_states = 5
            self.init_smoothing = True
            self.temporal_weights = torch.Tensor([0.1,0.1,0.15,0.25,0.4]).unsqueeze(0).unsqueeze(2).cuda() #Size (1,5,1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.InstanceNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))
        #Do not optimize the FAN
        for p in self.parameters():
            p.requires_grad = False


        if self.attention:
            n_in_features = 256*(num_modules+1) #Heatmap is applied hence no need to have it
        else:
            n_in_features = 256*(num_modules+1)+68 #68 for the heatmap
        
        n_features = [(256, 256)]*(n_blocks)

        self.emo_convs = []
        self.conv1x1_input_emo_2 =nn.Conv2d(n_in_features, 256, kernel_size=1, stride=1, padding=0)
        for in_f, out_f in n_features:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2,2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, self.n_expression + n_reg))

    def forward(self, x, reset_smoothing=False):
        
        #Resets the temporal smoothing
        if self.init_smoothing:
            self.init_smoothing = False
            self.temporal_state = torch.zeros(x.size(0), self.n_temporal_states, self.n_expression+self.n_reg).cuda()              
        if reset_smoothing:
            self.temporal_state = self.temporal_state.zeros_() 

        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg_features = []

        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            hg_features.append(ll)

        hg_features_cat = torch.cat(tuple(hg_features), dim=1)

        if self.attention:
            mask = torch.sum(tmp_out, dim=1, keepdim=True)
            hg_features_cat *= mask
            emo_feat = torch.cat((x, hg_features_cat), dim=1)
        else:
            emo_feat = torch.cat([x, hg_features_cat, tmp_out], dim=1)
        
        emo_feat_conv1D = self.conv1x1_input_emo_2(emo_feat)
        final_features = self.emo_net_2(emo_feat_conv1D)
        final_features = self.avg_pool_2(final_features)
        #batch_size = final_features.shape[0]
        #final_features = final_features.view(batch_size, final_features.shape[1])
        #final_features = self.emo_fc_2(final_features)
        
        #if self.temporal_smoothing:
        #    with torch.no_grad():
        #        self.temporal_state[:,:-1,:] = self.temporal_state[:,1:,:]
        #        self.temporal_state[:,-1,:] = final_features 
        #        final_features = torch.sum(self.temporal_weights*self.temporal_state, dim=1)

        #return {'heatmap': tmp_out, 'expression': final_features[:,:-2], 'valence': final_features[:,-2], 'arousal':final_features[:,-1]}
        return final_features  #così ritorna 256

  
    def eval(self):
        
        for module in self.children():
            module.eval()



#############################################################################

class VisitorNetInception(BaseModel):
    def __init__(self):
        super(VisitorNetInception, self).__init__()
        dr_rate = 0.5
        rnn_hidden_size = 256
        rnn_num_layers = 1
        i = 0
        num_features = 512
        baseModel = InceptionResnetV1(pretrained='vggface2')
        for child in baseModel.children():
          if i < 12:    #standard12
               for param in child.parameters():
                    param.requires_grad = False
          else:
               for param in child.parameters():
                    param.requires_grad = True
          i +=1
        # Example of using Sequential
        self.model = nn.Sequential(TimeWarp((baseModel), method='loop'),
                            nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True),
                            extractlastcell(),
                            nn.Dropout(dr_rate),
                            nn.Linear(256, 128),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(inplace=True),
                            nn.Dropout(dr_rate))
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

class VisitorNetEmonet(BaseModel):
    def __init__(self):
        super(VisitorNetEmonet, self).__init__()

        self.state_dict_path = 'model/pretrained/emonet_8.pth'

        dr_rate= 0.5
        rnn_hidden_size = 256
        rnn_num_layers = 1
        baseModel = EmoNet()
        baseModel.load_state_dict(torch.load(self.state_dict_path))
        
        i = 0
        for child in baseModel.children():
            if i < 28:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            i +=1

        num_features = 256

        # Example of using Sequential
        self.model = nn.Sequential(TimeWarp((baseModel), method='loop'),
                            nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True),
                            extractlastcell(),
                            nn.Dropout(dr_rate),
                            nn.Linear(256, 128),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(inplace=True),
                            nn.Dropout(dr_rate))
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

##################################################################################

class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=3):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

class VisitorNetResnet(BaseModel):
    def __init__(self):
        super(VisitorNetResnet, self).__init__()

        # Example of using Sequential
        self.model = nn.Sequential(ResCNNEncoder(),
        DecoderRNN())

    def forward(self, x):
        out = self.model(x)
        return out