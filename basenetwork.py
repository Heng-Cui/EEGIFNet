import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from thop import profile
from thop import clever_format
import time

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()

        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        self.fc4 = nn.Linear(512, 512)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.fc4(x)
        # x = self.relu4(x)
        # x = self.drop4(x)

        return x




class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()

        self.c1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.BatchNorm1d(num_features=64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)

        self.c2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU()

        self.c3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b3 = nn.BatchNorm1d(num_features=64)
        self.relu3 = nn.ReLU()

        self.c4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b4 = nn.BatchNorm1d(num_features=64)
        self.relu4 = nn.ReLU()

        self.f1 = nn.Linear(64 * 512, 512)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)
        x = self.drop1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.relu3(x)
        x = self.drop1(x)
        #
        x = self.c4(x)
        x = self.b4(x)
        x = self.relu4(x)
        x = self.drop1(x)

        out = x.view(x.size(0), -1)
        out = self.f1(out)

        return out



class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        self.f1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.f2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.f3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.unsqueeze(2)
        x, _ = self.rnn(x)

        x = x.reshape(x.size(0), -1)
        x = self.f1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.f2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.f3(x)


        return x



class Complex_CNN(nn.Module):
    def __init__(self):
        super(Complex_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.b1 = nn.BatchNorm1d(num_features=32)
        self.relu = nn.ReLU()

        self.layer1_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )

        self.layer2_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )

        self.layer1_5 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )

        self.layer2_5 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )

        self.layer1_7 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )

        self.layer2_7 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.b3 = nn.BatchNorm1d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(32 * 512 * 3, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.relu(x)
        x_emb = x
        x1_3_skip = x
        x1_5_skip = x
        x1_7_skip = x
        x1_3 = x1_3_skip + self.layer1_3(x_emb)
        x1_5 = x1_5_skip + self.layer1_5(x_emb)
        x1_7 = x1_7_skip + self.layer1_7(x_emb)
        x2_3_skip = x1_3
        x2_5_skip = x1_5
        x2_7_skip = x1_7
        x2_3 = x2_3_skip + self.layer2_3(x1_3)
        x2_5 = x2_5_skip + self.layer2_5(x1_5)
        x2_7 = x2_7_skip + self.layer2_7(x1_7)
        emb_x = torch.cat((x2_3, x2_5, x2_7), dim=-1)

        emb_x = self.conv3(emb_x)
        emb_x = self.b3(emb_x)
        emb_x = self.relu3(emb_x)
        emb_x = emb_x.view(emb_x.size(0), -1)
        emb_x = self.fc(emb_x)

        return emb_x




class Novel_CNN(nn.Module):
    def __init__(self):
        super(Novel_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool1d(2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool1d(2, stride=2)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool1d(2, stride=2)
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.AvgPool1d(2, stride=2)
        self.conv9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.AvgPool1d(2, stride=2)
        self.conv11 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv12 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.pool6 = nn.AvgPool1d(2, stride=2)
        self.conv13 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        self.conv14 = nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2048*8, 512)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool4(x)

        x = self.conv9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool5(x)

        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool6(x)

        x = self.conv13(x)
        x = self.relu(x)
        x = self.conv14(x)
        x = self.relu(x)
        x = self.drop(x)

        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out














if __name__ == '__main__':
    model = FCNN()
    # x = torch.randn((100, 1,1024))
    # y,s = model(x)
    # print(y.shape)
    # print(s.shape)

    input = torch.randn(1, 1,512)
    flops,params = profile(model, inputs=(input,))
    print(flops,params)
    flops, params = clever_format([flops,params], "%.3f")
    print(flops, params)

    x = torch.randn((1000, 1, 512))
    start = time.time()
    y = model(x)
    end = time.time()
    print(end-start)

