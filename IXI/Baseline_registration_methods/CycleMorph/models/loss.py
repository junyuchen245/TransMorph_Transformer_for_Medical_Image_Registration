import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dD = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])
        dH = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])
        dW = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])
        if(self.penalty == "l2"):
            dD = dD * dD
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dD) + torch.mean(dH) + torch.mean(dW)) / 3.0
        return loss

class crossCorrelation3D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9, 9), voxel_weights=None):
        super(crossCorrelation3D, self).__init__()
        self.in_ch = in_ch
        self.kernel = kernel
        self.voxel_weight = voxel_weights
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1], self.kernel[2]])).cuda()


    def forward(self, input, target):
        II = input * input
        TT = target * target
        IT = input * target

        pad = (int((self.kernel[0]-1)/2), int((self.kernel[1]-1)/2), int((self.kernel[2]-1)/2))
        T_sum = F.conv3d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv3d(input, self.filt, stride=1, padding=pad)
        TT_sum = F.conv3d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv3d(II, self.filt, stride=1, padding=pad)
        IT_sum = F.conv3d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1] * self.kernel[2]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        # cross = (I-Ihat)(J-Jhat)
        cross = IT_sum - Ihat*T_sum - That*I_sum + That*Ihat*kernelSize
        T_var = TT_sum - 2*That*T_sum + That*That*kernelSize
        I_var = II_sum - 2*Ihat*I_sum + Ihat*Ihat*kernelSize
        cc = cross*cross / (T_var*I_var+1e-5)

        loss = -1.0 * torch.mean(cc)
        return loss

class Grad3d_vxm(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d_vxm, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)