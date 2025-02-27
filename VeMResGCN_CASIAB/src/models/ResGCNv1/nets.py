import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .modules import ResGCN_Module

#########################view embedding###################################

class Temporal(nn.Module):
    def __init__(self, inplanes, planes, bias=False, **kwargs):     # inplanes = input channel, planes = output channel
        super(Temporal, self).__init__()

    def forward(self, x):

        out = torch.max(x, 2)[0]   # x = tensor, 2 = 2nd index vanished, [0] = convert into 0 dimension
        return out

def gem(x, p=6.5, eps=1e-6):

    return F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=6.5, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        # print('p-',self.p)
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

#########################################################################


class ResGCN_Input_Branch(nn.Module):
    def __init__(self, structure, block, num_channel, A, **kwargs):
        super(ResGCN_Input_Branch, self).__init__()

        self.register_buffer('A', A)

        module_list = [ResGCN_Module(num_channel, 64, 'Basic', A, initial=True, **kwargs)] # in=3, out=64   (1)-structure-init.py
        module_list += [ResGCN_Module(64, 64, 'Basic', A, initial=True, **kwargs) for _ in range(structure[0] - 1)]   # no entry
        module_list += [ResGCN_Module(64, 64, block, A, **kwargs) for _ in range(structure[1] - 1)]   # in=64, out=64   (2) er 0 index
        module_list += [ResGCN_Module(64, 32, block, A, **kwargs)]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        x = self.bn(x)
        for layer in self.layers:
            x = layer(x, self.A)

        return x


class ResGCN(nn.Module):
    def __init__(self, module, structure, block, num_input, num_channel, num_class, A, **kwargs):
        super(ResGCN, self).__init__()

        self.register_buffer('A', A)

        # input branches
        self.input_branches = nn.ModuleList([
            ResGCN_Input_Branch(structure, block, num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [module(32*num_input, 128, block, A, stride=2, **kwargs)]
        module_list += [module(128, 128, block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, block, A, stride=2, **kwargs)]
        module_list += [module(256, 256, block, A, **kwargs) for _ in range(structure[3] - 1)]

        ################################################## change ########################################
        # Block 3 (N2)
        module_list += [module(256, 128, block, A, **kwargs)]
        module_list += [module(128, 128, block, A, **kwargs)]
        ##################################################################################################

        self.main_stream = nn.ModuleList(module_list)

        ###############################view embedding####################################################
        self.trans_view=nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(17, 96, 96)))]*11)

        self.Gem = GeM()
        self.tempool = Temporal(32, 32)

        self.bin_numgl = [17*1]

        self.global_pooling1d = nn.AdaptiveAvgPool1d(1)
        self.global_pooling2d = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(96, 11)
        ###################################################################################

        # output
        self.fcn1 = nn.Linear(128, num_class)
        self.fcn2 = nn.Linear(96, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):

        # N, I, C, T, V = x.size()
        #print("x size ", x.size())



        # input branches
        x_cat = []
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:,i,:,:,:]))
        x = torch.cat(x_cat, dim=1)

        feature = x
        #print("feature", x.size())

        # main stream
        for layer in self.main_stream:
            x = layer(x, self.A)

############################## 1st branch ############################################

        out_feature = self.global_pooling2d(x)               # [128, 128, 1]
        #print("out_feature", out_feature.size())
        output1 = self.fcn1(out_feature.squeeze())              # [128, 128]
        #print("output1", output1.size())

######################################################################################

############################## 2nd branch ############################################

        x = feature
        ###########################view embedding#####################################
        x_temp = self.tempool(x)                        # [128, 128, 17]
        #print("2", x_temp.size())
        #print("feature_size() ", x_temp.size())
        x_global  = self.global_pooling1d(x_temp)       # [128, 128, 1]
        #print("3", x_global.size())
        n, c, _ = x_global.size()                       # n = batch_size, c = channel
        x_feature = x_global.view(n, c)                 # [128, 128]
        #print("4", x_feature.size())

        # view prediction
        angle_probe = self.cls(x_feature)               # [128, 11]
        #print("5", angle_probe.size())
        _, angle = torch.max(angle_probe, 1)            # [128]
        #print("6", angle.size())

        ############# HPP ######################
        x_temp = torch.unsqueeze(x_temp, 3)                     # [128, 128, 17, 1]
        #print("7", x_temp.size())
        _, c2d, _, _ = x_temp.size()                            # c2d = 128 (no of channel)
        feature = list()
        for num_bin in self.bin_numgl:
            z = x_temp.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            z2 = self.Gem(z).squeeze(-1)
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()      # [17, 128, 128]
        #print("8", feature.size())
        feature = feature.permute(1, 0, 2).contiguous()                   # [128, 17, 128]
        #print("9", feature.size())
        ############# End of HPP ######################

        #v_temp = x_temp.permute(0, 2, 1).contiguous()

        x_temp = feature
        feature_rt = []
        for j in range(x_temp.shape[0]):  # iteration on 256
          feature_now = ((x_temp[j].unsqueeze(1)).bmm(self.trans_view[angle[j]])).squeeze(1)
          feature_rt.append(feature_now)

        feature = torch.cat([x.unsqueeze(0) for x in feature_rt])  # [128, 17, 128]
        #print("10", feature.size())
        feature = feature.permute(0, 2, 1).contiguous()            # [128, 128, 17]
        #print("11", feature.size())

        #####################################################################################


        out_feature = self.global_pooling1d(feature)               # [128, 128, 1]
        #print("12", out_feature.size())
        output2 = self.fcn2(out_feature.squeeze())              # [128, 128]
        #print("output2", output2.size())

###############################################################################################

        final_out = torch.cat((output1, output2), 1)
        #print("final_out", final_out.size())

        # L2 normalization
        out_feature = F.normalize(final_out, dim=1, p=2)         # 128, 128

        return out_feature, angle_probe


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #m.bias = None
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    for m in modules:
        if isinstance(m, ResGCN_Module):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)
