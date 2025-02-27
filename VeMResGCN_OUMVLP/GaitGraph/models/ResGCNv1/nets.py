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

        module_list = [ResGCN_Module(num_channel, 64, 'Basic', A, initial=True, **kwargs)]
        module_list += [ResGCN_Module(64, 64, 'Basic', A, initial=True, **kwargs) for _ in range(structure[0] - 1)]
        module_list += [ResGCN_Module(64, 64, block, A, **kwargs) for _ in range(structure[1] - 1)]
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
            ResGCN_Input_Branch(structure, block, 5, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [module(32*num_input, 128, block, A, stride=2, **kwargs)]
        module_list += [module(128, 128, block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, block, A, stride=2, **kwargs)]
        module_list += [module(256, 256, block, A, **kwargs) for _ in range(structure[3] - 1)]


        # Block N2
        #module_list += [module(256, 128, block, A, **kwargs)]
        #module_list += [module(128, 128, block, A, **kwargs)]

        self.main_stream = nn.ModuleList(module_list)


        ###############################view embedding####################################################
        self.trans_view=nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(18, 96, 96)))]*14)

        self.Gem = GeM()
        self.tempool = Temporal(96, 96)

        self.bin_numgl = [18*1]

        self.global_pooling1d = nn.AdaptiveAvgPool1d(1)
        self.global_pooling2d = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(96, 14)
        ###################################################################################

        # output
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn1 = nn.Linear(256, num_class)
        self.fcn2 = nn.Linear(96, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):

        # (N, T, V, I = 3, 2 * C) => (N, I, C * 2, T, V)
        #print("X_values", x.size())

        x = x.permute(0, 3, 4, 1, 2)

        # input branches
        x_cat = []
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:, i]))

        x = torch.cat(x_cat, dim=1)

        feature = x
        #print("feature", feature.size())

#############################################    1st branch    ###################################################################
        # main stream
        for layer in self.main_stream:
            x = layer(x, self.A)
        f = x

        # output
        x = self.global_pooling(x)
        x = self.fcn1(x.squeeze())
        output1 = x
        #print("output1", output1.size())
##################################################################################################################################

#############################################    2nd branch    ###################################################################
        y = feature                   # x = [384, 96, 30, 18]
        #print("y", feature.size())

            ###########################view embedding#####################################
        x_temp = self.tempool(y)                        # [384, 96, 18]
        #print("x_temp", x_temp.size())

        x_global  = self.global_pooling1d(x_temp)       # [384, 96, 1]
        #print("x_global", x_global.size())
        n, c, _ = x_global.size()                       # n = batch_size, c = channel
        x_feature = x_global.view(n, c)                 # [384, 96]
        #print("x_feature", x_feature.size())


        # view prediction
        angle_probe = self.cls(x_feature)               # [384, 14]
        _, angle = torch.max(angle_probe, 1)            # [384]


        ############# HPP ######################
        x_temp = torch.unsqueeze(x_temp, 3)                        # [384, 256, 18, 1]
        _, c2d, _, _ = x_temp.size()                            # c2d = 128 (no of channel)  x = 384, 256, 8, 18
        feature = list()
        for num_bin in self.bin_numgl:
            z = x_temp.view(n, c2d, num_bin, -1).contiguous()      # 384, 256, 18, 1
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            z2 = self.Gem(z).squeeze(-1)     # gem = F.avg_pool2d     # 384, 256, 18
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()  # [18, 384, 256]
        feature = feature.permute(1, 0, 2).contiguous()                # [384, 18, 256]
        ############# End of HPP ######################

        #print("feature", feature.size())

        #x_temp = x_temp.permute(0, 3, 2, 1).contiguous()       # [384, 256, 8, 18] -> [384, 18, 8, 256]

        x_temp = feature     #384, 256, 8, 18
        #print("x_temp", x_temp.size())
        feature_rt = []

        for j in range(x_temp.shape[0]):  # iteration on batch 384
          feature_now = ((x_temp[j].unsqueeze(1)).bmm(self.trans_view[angle[j]])).squeeze(1)

          # 18, 128 -> 18, 8, 256          18, 256, 256
                 #     b   n   m           b    m   p
                 #    b n p = 18, 1, 256 -> 18, 256
          feature_rt.append(feature_now)

        feature = torch.cat([x.unsqueeze(0) for x in feature_rt])  # [384, 18, 8, 256]
        feature = feature.permute(0, 2, 1).contiguous()            # [384, 256, 8, 18]
        #print("feature", feature.size())

        #####################################################################################

        # output
        out_feature = self.global_pooling1d(feature)               # [384, 256, 1, 1]
        #print("out_feature AAAAAA",out_feature.size())
        out_feature = self.fcn2(out_feature.squeeze())              # [384, 256]
        #print("out_feature BBBBB", out_feature.size())
        output2 = out_feature
        #print("output2", output2.size())
##################################################################################################################################

        final_out = torch.cat((output1, output2), 1)
        #print("final out", final_out.size())

        # L2 normalization
        out_feature = F.normalize(final_out, dim=1, p=2)         # 384, 256
        #print("out_feature ", out_feature.size())
        return out_feature, f, angle_probe



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
