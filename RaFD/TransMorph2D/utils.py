import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class register_model(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

def dice_val(y_pred, y_true, num_clus):
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))

def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

import re
def process_label():
    #process labeling information for FreeSurfer
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255]


    file1 = open('label_info.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict

def write2csv(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def dice_val_substruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(46):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

def dice(y_pred, y_true, ):
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2.*intersection) / (union + 1e-5)
    return dsc

def smooth_seg(binary_img, sigma=1.5, thresh=0.4):
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img

def get_mc_preds(net, inputs, mc_iter: int = 25):
    """Convenience fn. for MC integration for uncertainty estimation.
    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: input to net
        mc_iter: number of MC samples
        post_processor: process output of net before computing loss (e.g. downsampler in SR)
        mask: multiply output and target by mask before computing loss (for inpainting)
    """
    img_list = []
    flow_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
    return img_list, flow_list

def calc_uncert(tar, img_list):
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar)**2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert

def calc_segs(seg, flows):
    segs = []
    reg_model = register_model((160, 192, 224), 'nearest')
    reg_model.cuda()
    for flow in flows:
        def_seg = reg_model([seg.cuda().float(), flow.cuda()])
        def_seg = def_seg.detach().cpu().numpy()[0, 0, :, :, :]
        segs.append(def_seg)
    return segs

def calc_error(tar, img_list):
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar)**2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert

def get_mc_preds_w_errors(net, inputs, target, mc_iter: int = 25):
    """Convenience fn. for MC integration for uncertainty estimation.
    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: input to net
        mc_iter: number of MC samples
        post_processor: process output of net before computing loss (e.g. downsampler in SR)
        mask: multiply output and target by mask before computing loss (for inpainting)
    """
    img_list = []
    flow_list = []
    MSE = nn.MSELoss()
    err = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            err.append(MSE(img, target).item())
    return img_list, flow_list, err

def get_diff_mc_preds(net, inputs, mc_iter: int = 25):
    """Convenience fn. for MC integration for uncertainty estimation.
    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: input to net
        mc_iter: number of MC samples
        post_processor: process output of net before computing loss (e.g. downsampler in SR)
        mask: multiply output and target by mask before computing loss (for inpainting)
    """
    img_list = []
    flow_list = []
    disp_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, _, flow, disp = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            disp_list.append(disp)
    return img_list, flow_list, disp_list

def uncert_regression_gal(img_list, reduction = 'mean'):
    img_list = torch.cat(img_list, dim=0)
    mean = img_list[:,:-1].mean(dim=0, keepdim=True)
    ale = img_list[:,-1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)
    #if epi.shape[1] == 3:
    epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == 'sum':
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()

def uceloss(errors, uncert, n_bins=15, outlier=0.0, range=None):
    device = errors.device
    if range == None:
        bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    prop_in_bin_list = []

    uce = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |uncertainty - error| in each bin
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n
        prop_in_bin_list.append(prop_in_bin)
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].float().mean()  # err()
            avg_uncert_in_bin = uncert[in_bin].mean()  # uncert()
            uce += torch.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin

            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=device)
    avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)
    prop_in_bin = torch.tensor(prop_in_bin_list, device=device)

    return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin