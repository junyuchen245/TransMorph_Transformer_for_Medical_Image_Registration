import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        #dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            #dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy)# + torch.mean(dz)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
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

class Grad3DiTV(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(Grad3DiTV, self).__init__()
        a = 1

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
        d = torch.mean(torch.sqrt(dx+dy+dz+1e-6))
        grad = d / 3.0
        return grad

class DisplacementRegularizer(torch.nn.Module):
    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:,i,...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return torch.mean(norms)/3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    def forward(self, disp, _):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy

class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def forward(self, y_pred, y_true):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
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

class MutualInfomation(torch.nn.Module):
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes
    (e.g. probabilistic segmentaitons)
    More information/citation:
    - Courtney K Guo.
      Multi-modal image registration with unsupervised deep learning.
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, JE Iglesias, B Fischl, AV Dalca.
      Learning image registration without images.
      arXiv preprint arXiv:2004.10282, 2020. https://arxiv.org/abs/2004.10282
    - https://github.com/adalca/neurite/blob/dev/neurite/tf/metrics.py
    Includes functions that can compute mutual information between volumes,
      between segmentations, or between a volume and a segmentation map
    mi = MutualInformation()
    mi.volumes
    mi.segs
    mi.volume_seg
    mi.channelwise
    mi.maps
    """
    def __init__(self, type='volumes', bin_centers=None, nb_bins=None, min_clip=None, max_clip=None, soft_bin_alpha=1):
        super(MutualInfomation, self).__init__()
        """
        Initialize the mutual information class
        Arguments below are related to soft quantizing of volumes, which is done automatically 
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise()) 
        using these parameters
        Args:
           bin_centers (np.float32, optional): array or list of bin centers. 
               Defaults to None.
           nb_bins (int, optional):  number of bins, if bin_centers is not specified. 
               Defaults to 16.
           min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
           max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
           soft_bin_alpha (int, optional): alpha in RBF of soft quantization. Defaults to 1.
        """
        self.type = type
        self.bin_centers = None
        if bin_centers is not None:
            self.bin_centers = torch.from_numpy(bin_centers).cuda().float()
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins
        if bin_centers is None and nb_bins is None:
            self.nb_bins = 16

        self.min_clip = min_clip
        if self.min_clip is None:
            self.min_clip = -np.inf

        self.max_clip = max_clip
        if self.max_clip is None:
            self.max_clip = np.inf

        self.soft_bin_alpha = soft_bin_alpha

    def volumes(self, x, y):
        """
        Mutual information for each item in a batch of volumes.
        Algorithm:
        - use neurite.utils.soft_quantize() to create a soft quantization (binning) of
          intensities in each channel
        - channelwise()
        Parameters:
            x and y:  [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = x.shape[1]
        tensor_channels_y = y.shape[1]
        msg = 'volume_mi requires two single-channel volumes. See channelwise().'
        assert tensor_channels_x == 1, msg
        assert tensor_channels_y == 1, msg

        # volume mi
        return torch.flatten(self.channelwise(x, y))

    def segs(self, x, y):
        """
        Mutual information between two probabilistic segmentation maps.
        Wraps maps()
        Parameters:
            x and y:  [bs, nb_labels, ...]
        Returns:
            Tensor of size [bs]
        """
        # volume mi
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """
        Mutual information between a volume and a probabilistic segmentation maps.
        Wraps maps()
        Parameters:
            x and y: a volume and a probabilistic (soft) segmentation. Either:
              - x: [bs, ..., 1] and y: [bs, ..., nb_labels], Or:
              - x: [bs, ..., nb_labels] and y: [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = x.shape[1]
        tensor_channels_y = y.shape[1]
        msg = 'volume_seg_mi requires one single-channel volume.'
        assert min(tensor_channels_x, tensor_channels_y) == 1, msg
        msg = 'volume_seg_mi requires one multi-channel segmentation.'
        assert max(tensor_channels_x, tensor_channels_y) > 1, msg

        # transform volume to soft-quantized volume
        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[:, 0, ...])  # [bs, B, ...]
        else:
            y = self._soft_sim_map(y[:, 0, ...])  # [bs, B, ...]

        return self.maps(x, y)  # [bs]

    def channelwise(self, x, y):
        """
        Mutual information for each channel in x and y. Thus for each item and channel this
        returns retuns MI(x[...,i], x[...,i]). To do this, we use neurite.utils.soft_quantize() to
        create a soft quantization (binning) of the intensities in each channel
        Parameters:
            x and y:  [bs, ..., C]
        Returns:
            Tensor of size [bs, C]
        """
        # check shapes
        tensor_shape_x = x.shape
        tensor_shape_y = y.shape
        assert tensor_shape_x == tensor_shape_y, 'volume shapes do not match'
        # reshape to [bs, V, C]
        if len(tensor_shape_x) != 3:
            x = torch.reshape(x, (tensor_shape_x[0], tensor_shape_x[1], -1))  # [bs, C, V]
            x = x.permute(0, 2, 1)# [bs, V, C]
            y = torch.reshape(y, (tensor_shape_x[0], tensor_shape_x[1], -1))  # [bs, C, V]
            y = y.permute(0, 2, 1)  # [bs, V, C]

        # move channels to first dimension
        cx = x.permute(2, 0, 1) # [C, bs, V]
        cy = y.permute(2, 0, 1) # [C, bs, V]

        # soft quantize
        cxq = self._soft_sim_map(cx)  # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)  # [C, bs, V, B]
        # get mi
        cout = []
        for i in range(cxq.shape[0]):
            cout.append(self.maps(cxq[i:i+1, ...], cyq[i:i+1, ...]))
        cout = torch.stack(cout, dim=0) # [C, bs]

        # permute back
        return cout.permute(1, 0) # [bs, C]

    def maps(self, x, y):
        """
        Computes mutual information for each entry in batch, assuming each item contains
        probability or similarity maps *at each voxel*. These could be e.g. from a softmax output
        (e.g. when performing segmentaiton) or from soft_quantization of intensity image.
        Note: the MI is computed separate for each itemin the batch, so the joint probabilities
        might be  different across inputs. In some cases, computing MI actoss the whole batch
        might be desireable (TODO).
        Parameters:
            x and y are probability maps of size [bs, ..., B], where B is the size of the
              discrete probability domain grid (e.g. bins/labels). B can be different for x and y.
        Returns:
            Tensor of size [bs]
        """

        # check shapes
        tensor_shape_x = x.shape
        tensor_shape_y = y.shape
        assert tensor_shape_x == tensor_shape_y, 'volume shapes do not match'
        assert torch.min(x) >= 0, 'voxel values must be non-negative'
        assert torch.min(y) >= 0, 'voxel values must be non-negative'

        eps = 1e-6

        # reshape to [bs, V, B]
        if len(tensor_shape_x) != 3:
            x = torch.reshape(x, (tensor_shape_x[1], tensor_shape_x[2], tensor_shape_x[3])) # [bs, V, B1]
            y = torch.reshape(y, (tensor_shape_x[1], tensor_shape_x[2], tensor_shape_x[3])) # [bs, V, B2]

        # x probability for each batch entry
        px = torch.sum(x, 1, keepdim=True)  # [bs, 1, B1]
        px = px / torch.sum(px, dim=2, keepdim=True)
        # y probability for each batch entry
        py = torch.sum(y, 1, keepdim=True)  # [bs, 1, B2]
        py = py / torch.mean(py, dim=2, keepdim=True)

        # joint probability for each batch entry
        x_trans = x.permute(0, 2, 1)  # [bs, B1, V]
        pxy = torch.bmm(x_trans, y)  # [bs, B1, B2]
        pxy = pxy / (torch.sum(pxy, dim=[1, 2], keepdim=True) + eps)  # [bs, B1, B2]

        # independent xy probability
        px_trans = px.permute(0, 2, 1)  # [bs, B1, 1]
        pxpy = torch.bmm(px_trans, py)  # [bs, B1, B2]
        pxpy_eps = pxpy + eps

        # mutual information
        log_term = torch.log(pxy / pxpy_eps + eps)  # [bs, B1, B2]
        mi = torch.sum(pxy * log_term, dim=[1, 2])  # [bs]
        return mi

    def _soft_log_sim_map(self, x):
        """
        soft quantization of intensities (values) in a given volume
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image.
        Returns:
            volume with one more dimension [bs, ..., B]
        """

        return self.soft_quantize(x,
                                  alpha=self.soft_bin_alpha,
                                  bin_centers=self.bin_centers,
                                  nb_bins=self.nb_bins,
                                  min_clip=self.min_clip,
                                  max_clip=self.max_clip,
                                  return_log=True)  # [bs, ..., B]

    def _soft_sim_map(self, x):
        """
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image.
        Returns:
            volume with one more dimension [bs, ..., B]
        """
        return self.soft_quantize(x,
                                  alpha=self.soft_bin_alpha,
                                  bin_centers=self.bin_centers,
                                  nb_bins=self.nb_bins,
                                  min_clip=self.min_clip,
                                  max_clip=self.max_clip,
                                  return_log=False)  # [bs, ..., B]

    def _soft_prob_map(self, x, **kwargs):
        """
        normalize a soft_quantized volume at each voxel, so that each voxel now holds a prob. map
        Parameters:
            x [bs, ..., B]: soft quantized volume
        Returns:
            x [bs, ..., B]: renormalized so that each voxel adds to 1 across last dimension
        """
        eps = 1e-6
        x_hist = self._soft_sim_map(x, **kwargs)  # [bs, ..., B]
        x_hist_sum = torch.sum(x_hist, -1, keepdim=True) + eps  # [bs, ..., B]
        x_prob = x_hist / x_hist_sum  # [bs, ..., B]
        return x_prob

    def soft_quantize(self, x,
                      bin_centers=None,
                      nb_bins=16,
                      alpha=1,
                      min_clip=-np.inf,
                      max_clip=np.inf,
                      return_log=False):
        """
        (Softly) quantize intensities (values) in a given volume, based on RBFs.
        In numpy this (hard quantization) is called "digitize".

        Code modified based on:
        https://github.com/adalca/neurite/blob/3858b473fcdc89354fe645a453d75ad01c794c8a/neurite/tf/utils/utils.py#L860
        """
        if bin_centers is not None:
            if not torch.is_tensor(bin_centers):
                bin_centers = torch.from_numpy(bin_centers).cuda().float()
            else:
                bin_centers = bin_centers.cuda().float()
            #assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]
        else:
            if nb_bins is None:
                nb_bins = 16
            # get bin centers dynamically
            minval = torch.min(x)
            maxval = torch.max(x)
            bin_centers = torch.linspace(minval.item(), maxval.item(), nb_bins)
        #print(bin_centers)

        # clipping at bin values
        x = x[..., None]  # [..., 1]
        x = torch.clamp(x, min_clip, max_clip)

        # reshape bin centers to be (1, 1, .., B)
        new_shape = [1] * (len(x.shape) - 1) + [nb_bins]
        bin_centers = torch.reshape(bin_centers, new_shape)  # [1, 1, ..., B]

        # compute image terms
        bin_diff = torch.square(x - bin_centers.cuda())  # [..., B]
        log = -alpha * bin_diff  # [..., B]

        if return_log:
            return log  # [..., B]
        else:
            return torch.exp(log)  # [..., B]

    def forward(self, y_pred, y_true):
        if self.type.lower() == 'volumes':
            mi = self.volumes(y_pred, y_true)
        elif self.type.lower() == 'segmentation':
            mi = self.segs(y_pred, y_true)
        elif self.type.lower() == 'volume segmentation':
            mi = self.volume_seg(y_pred, y_true)
        elif self.type.lower() == 'channelwise':
            mi = self.channelwise(y_pred, y_true)
        else:
            raise Exception("Type not implemented!")
        return -mi.mean()

class MIND_loss(torch.nn.Module):
    """
        Local (over window) normalized cross correlation loss.
        """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)



