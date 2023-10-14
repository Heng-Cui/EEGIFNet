import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torch.nn as nn
# Author: Haoming Zhang
#The code here not only include data importing, but also data standardization and the generation of analog noise signals

class CCA_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device).double()
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device).double()
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0
        #
        # torch.where(torch.isnan(D1),torch.full_like(D1,0), D1)
        # torch.where(torch.isnan(D2), torch.full_like(D2, 0), D2)
        # torch.where(torch.isnan(V1), torch.full_like(V1, 0), V1)
        # torch.where(torch.isnan(V2), torch.full_like(V2, 0), V2)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        # print(posInd1.size())
        # print(posInd2.size())

        a1 = torch.matmul(V1, torch.diag(D1 ** -0.5))
        SigmaHat11RootInv = torch.matmul(a1, V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())
        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0]).double()*r1).to(self.device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr

# class CCA_loss(nn.Module):
#     def __init__(self, outdim_size, use_all_singular_values, device):
#         super().__init__()
#         self.outdim_size = outdim_size
#         self.use_all_singular_values = use_all_singular_values
#         self.device = device
#         # print(device)
#
#     def forward(self, H1, H2):
#         """
#
#         It is the loss function of CCA as introduced in the original paper. There can be other formulations.
#
#         """
#
#         r1 = 1e-4
#         r2 = 1e-4
#         eps = 1e-12
#
#         H1, H2 = H1.t(), H2.t()
#         # assert torch.isnan(H1).sum().item() == 0
#         # assert torch.isnan(H2).sum().item() == 0
#
#         o1 = o2 = H1.size(0)
#         # print("H1.size(0)",H1.size(0))
#
#         m = H1.size(1)
#         #         print(H1.size())
#         # print("H1.size(1)",H1.size(1))
#         H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
#         H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
#         # H1bar = H1
#         # H2bar = H2
#         # assert torch.isnan(H1bar).sum().item() == 0
#         # assert torch.isnan(H2bar).sum().item() == 0
#
#         SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
#         SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
#         SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)
#         # assert torch.isnan(SigmaHat11).sum().item() == 0
#         # assert torch.isnan(SigmaHat12).sum().item() == 0
#         # assert torch.isnan(SigmaHat22).sum().item() == 0
#
#         # Calculating the root inverse of covariance matrices by using eigen decomposition
#         [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)  # D1 特征值 V1 特征向量
#         [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
#         # assert torch.isnan(D1).sum().item() == 0
#         # assert torch.isnan(D2).sum().item() == 0
#         # assert torch.isnan(V1).sum().item() == 0
#         # assert torch.isnan(V2).sum().item() == 0
#
#         # Added to increase stability
#         posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
#         D1 = D1[posInd1]
#         V1 = V1[:, posInd1]
#         posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
#         D2 = D2[posInd2]
#         V2 = V2[:, posInd2]
#         # print(posInd1.size())
#         # print(posInd2.size())
#
#         SigmaHat11RootInv = torch.matmul(
#             torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
#         SigmaHat22RootInv = torch.matmul(
#             torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())
#
#         Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
#         #         print(Tval.size())
#
#         if self.use_all_singular_values:
#             # all singular values are used to calculate the correlation
#             tmp = torch.matmul(Tval.t(), Tval)
#             corr = torch.trace(torch.sqrt(tmp))
#
#             # assert torch.isnan(corr).item() == 0
#         else:
#             # just the top self.outdim_size singular values are used
#             trace_TT = torch.matmul(Tval.t(), Tval)
#             trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0]) * r1).to(
#                 self.device))  # regularization for more stability
#             U, V = torch.symeig(trace_TT, eigenvectors=True)
#             U = torch.where(U > eps, U, (torch.ones(U.shape).float() * eps).to(self.device))
#             U = U.topk(self.outdim_size)[0]
#             corr = torch.sum(torch.sqrt(U))
#             '''
#             UHat,S,VHat = torch.svd(Tval)
#             UR = torch.matmul(SigmaHat11RootInv,UHat)
#
#             VR = torch.matmul(SigmaHat22RootInv,VHat)
#             '''
#         return -corr


class my_dataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data # ndarray
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        # when iterating on dataloader, a (data, labels) pair in type of tensor should be returned
        data = torch.from_numpy(self.data[index, :]).reshape(1, self.data.shape[1]).float()
        labels = torch.from_numpy(self.labels[index, :]).float()
        # without float(), the tensor element will be type double, causing error in forward()

        return data, labels

    def __len__(self):
        return self.data.shape[0]

class my_dataset_SNR(Dataset):

    def __init__(self, data, labels, snr):
        self.data = data # ndarray
        self.labels = labels
        self.snr = snr


    def __getitem__(self, idx):
        return self.data[idx].reshape(1, self.data.shape[1]), self.labels[idx], self.snr[idx]

    # def __getitem__(self, index):
    #     # when iterating on dataloader, a (data, labels) pair in type of tensor should be returned
    #     data = torch.from_numpy(self.data[index, :]).reshape(1, self.data.shape[1]).float()
    #     labels = torch.from_numpy(self.labels[index, :]).float()
    #     snr = torch.Tensor(self.snr[index]).float()
    #     # without float(), the tensor element will be type double, causing error in forward()
    #
    #     return data, labels, snr

    def __len__(self):
        return self.data.shape[0]

def cal_ACC_npy(predict, truth):
    # 计算ACC
    vy_ = predict - np.mean(predict, axis=1).reshape((predict.shape[0], 1))
    vy = truth - np.mean(truth, axis=1).reshape((truth.shape[0], 1))
    cc = np.sum(vy_ * vy, axis=-1) / (
            np.sqrt(np.sum(vy_ ** 2, axis=-1)) * np.sqrt(np.sum(vy ** 2, axis=-1)) + 1e-8)
    # print("cc:" + str(torch.mean(cc)))
    average_cc = np.mean(cc)
    return average_cc

def cal_ACC_tensor(predict, truth):
    vy_ = predict - torch.mean(predict, dim=1).unsqueeze(1)
    vy = truth - torch.mean(truth, dim=1).unsqueeze(1)
    cc = torch.sum(vy_ * vy, dim=-1) / (
            torch.sqrt(torch.sum(vy_ ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1)) + 1e-8)
    # print("cc:" + str(torch.mean(cc)))
    average_cc = torch.mean(cc)
    return average_cc

def ACCLoss(predict, truth):
    vy_ = predict - torch.mean(predict, dim=1).unsqueeze(1)
    vy = truth - torch.mean(truth, dim=1).unsqueeze(1)
    cc = torch.sum(vy_ * vy, dim=-1) / (
            torch.sqrt(torch.sum(vy_ ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1)) + 1e-8)
    # print("cc:" + str(torch.mean(cc)))
    average_cc = torch.mean(cc)
    return torch.ones_like(average_cc) -average_cc

def cal_RRMSE_tensor(predict, truth):
    l1 = (predict - truth) ** 2
    lo = torch.sum(l1, dim=1) / predict.shape[-1]
    l3 = (truth) ** 2
    l4 = torch.sum(l3, dim=1) / predict.shape[-1]
    rrmse = torch.sqrt(lo) / torch.sqrt(l4)
    rrmse = torch.mean(rrmse)
    return rrmse

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=1)  # power of noise
    ratio = PS / PN
    snr = np.mean(10 * np.log10(ratio))
    return snr

# A = np.ones((16, 512))
# B = np.ones((16, 512))
# c = np.sum(A**2, axis=-1)
# print(c.shape)
# A = torch.ones((16, 512))
# B = torch.ones((16, 512))
# c = cal_ACC_tensor(A, B)
# c = torch.mean(A, dim=1)
# print(c)
