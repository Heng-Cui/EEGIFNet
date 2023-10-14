import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Conv_Block(nn.Module):
    def __init__(self, channel=64, kernel=9, stride=1, padding=4):
        super(Conv_Block, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv1d(channel, channel // 2, kernel, stride, padding, bias=False),
            nn.BatchNorm1d(channel // 2),
            nn.Dropout(0.1),
            nn.Sigmoid(),
        )
        self.lay2 = nn.Sequential(
            nn.Conv1d(channel, channel, kernel, stride, padding),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(channel, channel, kernel, stride, padding),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            #nn.Conv1d(channel, channel // 2, kernel_size=1, stride=1),
            nn.Conv1d(channel, channel // 2, kernel, stride, padding),
            #nn.BatchNorm1d(channel // 2),
            #nn.Dropout(0.1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.lay2(x)


class Interaction_Block(nn.Module):
    def __init__(self, channel=64,outchannel=8):
        super(Interaction_Block, self).__init__()
        self.Conv_n2s = Conv_Block(channel*2)  # TODO 这里千万要注意，因为这里是两条路，一个n2s，一个s2n，所以是连个独立的卷积层
        self.Conv_s2n = Conv_Block(channel*2)  # TODO 1*1卷积核是不会学习到任何特征的，它只能起调整通道数的作用
        self.lay_s = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=outchannel, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.lay_n = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=outchannel, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    def forward(self, F_RA_s, F_RA_n):  # F_RA：[B, T, F', C]
        F_cat = torch.cat((F_RA_n, F_RA_s), dim=1)  # F_cat: [B, T, F', 2*C]

        Mask_n = self.Conv_n2s(F_cat)    # [B, C, T, F']
        Mask_s = self.Conv_s2n(F_cat)    # [B, C, T, F']

        H_n2s = F_RA_n * Mask_n   # [B, T, F', C]
        H_s2n = F_RA_s * Mask_s   # [B, T, F', C]

        # F_RA_S = F_RA_s + H_n2s   # [B, T, F', C]
        # F_RA_N = F_RA_n + H_s2n   # [B, T, F', C]

        H_n2s = self.lay_s(H_n2s)
        H_s2n = self.lay_n(H_s2n)
        F_RA_S = torch.cat((F_RA_s, H_n2s), dim = 1)   # [B, T, F', C]
        F_RA_N = torch.cat((F_RA_n, H_s2n), dim = 1) # [B, T, F', C]

        return F_RA_S, F_RA_N

# class Exchange(nn.Module):
#     def __init__(self):
#         super(Exchange, self).__init__()
#
#     def forward(self, e,n, bn_e, bn_n, bn_threshold=0):
#         bn1, bn2 = bn_e.weight.abs(), bn_n.weight.abs()
#         x1, x2 = torch.zeros_like(e), torch.zeros_like(n)
#         #print(x1.shape)
#         x1[:, bn1 >= bn_threshold] = e[:, bn1 >= bn_threshold]
#         x1[:, bn1 < bn_threshold] = n[:, bn1 < bn_threshold]
#         x2[:, bn2 >= bn_threshold] = n[:, bn2 >= bn_threshold]
#         x2[:, bn2 < bn_threshold] = e[:, bn2 < bn_threshold]
#         return x1, x2

class Exchange(nn.Module):
    def __init__(self, K=5):
        super(Exchange, self).__init__()
        self.K = K
    def forward(self, e,n, bn_e, bn_n, bn_threshold=0.01):
        bn1, bn2 = bn_e.weight.abs(), bn_n.weight.abs()
        print(bn2)
        _, idx1 = bn1.topk(self.K, largest=False, sorted=False)
        #print(idx1)
        _, idx2 = bn2.topk(self.K, largest=False, sorted=False)
        x1, x2 = torch.zeros_like(e), torch.zeros_like(n)
        #print(x1.shape)
        x1[:] = e[:]
        x1[:, idx1] = n[:, idx2]
        x2[:] = n[:]
        x2[:, idx2] = e[:, idx1]
        return x1, x2



class MA_MNet(nn.Module):
    def __init__(self):
        super(MA_MNet, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=9, stride=1, padding=4),
            nn.Sigmoid(),
        )

        self.fc = nn.Linear(32 * 512, 512)
    def forward(self, x, x1, x2):
        #out = torch.cat((x1, x.squeeze() - x2), dim=-1)
        x1 = x1.unsqueeze(1)
        x2 = x - x2.unsqueeze(1)
        mask = torch.cat((x, x1, x2), dim=1)
        mask = self.lay1(mask)
        #torch.ones_like(mask)
        out = x1*mask + x2*(1-mask)
        #F_RA_n * Mask_n


        return out.squeeze()

class OA_MNet(nn.Module):
    def __init__(self):
        super(OA_MNet, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=9, stride=1, padding=4),
            nn.Sigmoid(),
        )

        self.fc = nn.Linear(32 * 512, 512)
    def forward(self, x, x1, x2):
        #out = torch.cat((x1, x.squeeze() - x2), dim=-1)
        x1 = x1.unsqueeze(1)
        x2 = x - x2.unsqueeze(1)
        mask = torch.cat((x, x1, x2), dim=1)
        mask = self.lay1(mask)
        #torch.ones_like(mask)
        out = x1*mask + x2*(1-mask)
        #F_RA_n * Mask_n


        return out.squeeze()


class OA_INet(nn.Module):
    def __init__(self):
        super(OA_INet, self).__init__()

        self.c1_e = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.batnorm1_e = nn.BatchNorm1d(num_features=32)
        self.relu1_e = nn.ReLU()
        self.drop1_e = nn.Dropout(p=0.1)
        self.drop = nn.Dropout(p=0.1)

        self.c1_n = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.batnorm1_n = nn.BatchNorm1d(num_features=32)
        self.relu1_n = nn.ReLU()
        self.drop1_n = nn.Dropout(p=0.1)
        self.ex1 = Exchange()
        self.ex2 = Exchange()
        self.ex3 = Exchange()
        self.i1 = Interaction_Block(32)

        self.c2_e = nn.Conv1d(in_channels=32+8, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.batnorm2_e = nn.BatchNorm1d(num_features=32)
        self.relu2_e = nn.ReLU()

        self.c2_n = nn.Conv1d(in_channels=32+8, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.batnorm2_n = nn.BatchNorm1d(num_features=32)
        self.relu2_n = nn.ReLU()

        self.i2 = Interaction_Block(32)


        self.c3_e = nn.Conv1d(in_channels=32+8, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.batnorm3_e = nn.BatchNorm1d(num_features=32)
        self.relu3_e = nn.ReLU()

        self.c3_n = nn.Conv1d(in_channels=32+8, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.batnorm3_n = nn.BatchNorm1d(num_features=32)
        self.relu3_n = nn.ReLU()

        self.i3 = Interaction_Block(32)

        self.rnn_e = nn.GRU(input_size=32+8, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_n = nn.GRU(input_size=32+8, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

        self.i4 = Interaction_Block(64)

        self.f1_e = nn.Linear(64 * 512, 512)
        self.f1_n = nn.Linear(64 * 512, 512)
        #
        self.fc1 = nn.Linear(512*2, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)


    def forward(self, x):
        e = self.c1_e(x)
        e = self.batnorm1_e(e)

        n = self.c1_n(x)
        n = self.batnorm1_n(n)


        e = self.relu1_e(e)
        e = self.drop1_e(e)

        n = self.relu1_n(n)
        n = self.drop1_n(n)

        #e, n = self.ex1(e, n, self.batnorm1_e, self.batnorm1_n)
        e, n = self.i1(e, n)

        e = self.c2_e(e)
        e = self.batnorm2_e(e)

        n = self.c2_n(n)
        n = self.batnorm2_n(n)

        e = self.relu2_e(e)
        e = self.drop1_e(e)
        n = self.relu2_n(n)
        n = self.drop1_n(n)

        #e, n = self.ex2(e, n, self.batnorm2_e, self.batnorm2_n)
        e, n = self.i2(e, n)

        e = self.c3_e(e)
        e = self.batnorm3_e(e)
        n = self.c3_n(n)
        n = self.batnorm3_n(n)
        e = self.relu3_e(e)
        e = self.drop1_e(e)
        n = self.relu3_n(n)
        n = self.drop1_n(n)

        #e, n = self.ex3(e, n, self.batnorm3_e, self.batnorm3_n)
        e, n = self.i3(e, n)

        e, _ = self.rnn_e(e.permute(0, 2, 1))
        n, _ = self.rnn_n(n.permute(0, 2, 1))

        #e, n = self.i4(e.permute(0, 2, 1), n.permute(0, 2, 1))

        e = e.reshape(e.size(0), -1)
        n = n.reshape(n.size(0), -1)

        e = self.drop(e)
        n = self.drop(n)

        e_out = self.f1_e(e)
        n_out = self.f1_n(n)

        e_out = self.relu1(e_out)
        n_out = self.relu1(n_out)
        e_out = self.drop(e_out)
        n_out = self.drop(n_out)
        e_out = self.fc2(e_out)
        n_out = self.fc3(n_out)
        e_out = self.relu1(e_out)
        n_out = self.relu1(n_out)
        e_out = self.drop(e_out)
        n_out = self.drop(n_out)
        e_out = self.fc4(e_out)
        n_out = self.fc5(n_out)

        return e_out, n_out

class MA_INet(nn.Module):
    def __init__(self):
        super(MA_INet, self).__init__()

        self.c1_e = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm1_e = nn.BatchNorm1d(num_features=32)
        self.relu1_e = nn.ReLU()
        self.drop1_e = nn.Dropout(p=0.1)
        self.drop = nn.Dropout(p=0.1)

        self.c1_n = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm1_n = nn.BatchNorm1d(num_features=32)
        self.relu1_n = nn.ReLU()
        self.drop1_n = nn.Dropout(p=0.1)
        self.ex1 = Exchange()
        self.ex2 = Exchange()
        self.ex3 = Exchange()
        self.i1 = Interaction_Block(32)

        self.c2_e = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm2_e = nn.BatchNorm1d(num_features=32)
        self.relu2_e = nn.ReLU()

        self.c2_n = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm2_n = nn.BatchNorm1d(num_features=32)
        self.relu2_n = nn.ReLU()

        self.i2 = Interaction_Block(32)


        self.c3_e = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm3_e = nn.BatchNorm1d(num_features=32)
        self.relu3_e = nn.ReLU()

        self.c3_n = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm3_n = nn.BatchNorm1d(num_features=32)
        self.relu3_n = nn.ReLU()

        self.i3 = Interaction_Block(32)

        self.rnn_e = nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_n = nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

        self.i4 = Interaction_Block(64)

        self.f1_e = nn.Linear(64 * 64, 512)
        self.f1_n = nn.Linear(64 * 64, 512)
        #
        self.fc1 = nn.Linear(512*2, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)


    def forward(self, x):
        e = self.c1_e(x)
        e = self.batnorm1_e(e)

        n = self.c1_n(x)
        n = self.batnorm1_n(n)


        e = self.relu1_e(e)
        e = self.drop1_e(e)

        n = self.relu1_n(n)
        n = self.drop1_n(n)

        #e, n = self.ex1(e, n, self.batnorm1_e, self.batnorm1_n)
        #e, n = self.i1(e, n)

        e = self.c2_e(e)
        e = self.batnorm2_e(e)

        n = self.c2_n(n)
        n = self.batnorm2_n(n)

        e = self.relu2_e(e)
        e = self.drop1_e(e)
        n = self.relu2_n(n)
        n = self.drop1_n(n)

        #e, n = self.ex2(e, n, self.batnorm2_e, self.batnorm2_n)
        #e, n = self.i2(e, n)

        e = self.c3_e(e)
        e = self.batnorm3_e(e)
        n = self.c3_n(n)
        n = self.batnorm3_n(n)
        e = self.relu3_e(e)
        e = self.drop1_e(e)
        n = self.relu3_n(n)
        n = self.drop1_n(n)

        #e, n = self.ex3(e, n, self.batnorm3_e, self.batnorm3_n)
        #e, n = self.i3(e, n)

        e, _ = self.rnn_e(e.permute(0, 2, 1))
        n, _ = self.rnn_n(n.permute(0, 2, 1))

        #e, n = self.i4(e.permute(0, 2, 1), n.permute(0, 2, 1))

        e = e.reshape(e.size(0), -1)
        n = n.reshape(n.size(0), -1)

        e = self.drop(e)
        n = self.drop(n)

        e_out = self.f1_e(e)
        n_out = self.f1_n(n)

        e_out = self.relu1(e_out)
        #n_out = self.relu1(n_out)
        e_out = self.drop(e_out)
        #n_out = self.drop(n_out)
        e_out = self.fc2(e_out)
        #n_out = self.fc3(n_out)
        e_out = self.relu1(e_out)
        #n_out = self.relu1(n_out)
        e_out = self.drop(e_out)
        #n_out = self.drop(n_out)
        e_out = self.fc4(e_out)
        #n_out = self.fc5(n_out)

        return e_out, n_out


if __name__ == '__main__':
    a = torch.tensor([2,9,5,7,3,5,91,3,6,4])
    _,c = a.topk(4, largest=False, sorted=False)
    print(c)
    # s = torch.randn(3, 1, 512)
    # n = torch.randn(2, 64, 128)
    # model = TSIMNet()
    # S, _, N = model(s)
    # print(S.shape, N.shape)