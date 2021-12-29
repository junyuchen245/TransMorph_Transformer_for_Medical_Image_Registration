import math
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter

def read_txt_landmarks(filename, if_down=False):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    landmarks = []
    for line in Lines:
        line = line.strip().split('\t')
        #print("Line: {}".format(line))
        if if_down:
            lm = [int(int(line[0])/2), int(int(line[1])/2), int(line[2])]
        else:
            lm = [int(line[0]), int(line[1]), int(line[2])]
        landmarks.append(lm)
    return landmarks

def deform_landmarks(source_lms, target_lms, flow, if_down=False):
    factor = 1
    if if_down:
        factor = 2
    u = flow[0, :, :, :]
    v = flow[1, :, :, :]
    w = flow[2, :, :, :]
    pixel_wth = [2.5, 0.97*factor, 0.97*factor]
    flow_fields = [u, v, w]
    diff_all = []
    raw_diff_all = []
    for i in range(len(source_lms)):
        source_lm = source_lms[i]
        target_lm = target_lms[i]
        diff = 0
        raw_diff = 0
        for j in range(len(source_lm)):
            sor_pnt = source_lm[2-j]
            tar_pnt = target_lm[2-j]
            def_field = flow_fields[j]
            out_pnt = sor_pnt - def_field[source_lm[2]-1, source_lm[1]-1, source_lm[0]-1]
            diff += (np.abs(out_pnt-tar_pnt)*pixel_wth[j])**2
            raw_diff += (np.abs(sor_pnt-tar_pnt)*pixel_wth[j])**2
        diff_all.append(math.sqrt(diff))
        raw_diff_all.append(math.sqrt(raw_diff))
    return np.mean(np.array(diff_all)), np.mean(np.std(diff_all)), np.mean(np.array(raw_diff_all)), np.std(np.array(raw_diff_all))

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

def dice_val_CTsubstruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=16)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=16)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(16):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

import re
def process_label():
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

def process_CT_label():
    seg_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    seg_name = ['Body-Outline', 'Bone-Structure', 'Right-Lung', 'Left-Lung', 'Heart', 'Liver', 'Spleen', 'Right-Kidney',
                'Left-Kidney', 'Stomach', 'Pancreas', 'Large-Intestine', 'Prostate', 'Bladder', 'Gall-Bladder', 'Thyroid']
    return seg_name

def jacobian_determinant(disp):
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
    volshape = disp.shape[1:]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, 0)
    #print(grid)
    #sys.exit(0)

    # compute gradients
    [xFX, xFY, xFZ] = np.gradient(grid[0] - disp[0])
    [yFX, yFY, yFZ] = np.gradient(grid[1] - disp[1])
    [zFX, zFY, zFZ] = np.gradient(grid[2] - disp[2])

    jac_det = np.zeros(grid[0].shape)
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            for k in range(grid.shape[3]):
                jac_mij = [[xFX[i, j, k], xFY[i, j, k], xFZ[i, j, k]], [yFX[i, j, k], yFY[i, j, k], yFZ[i, j, k]], [zFX[i, j, k], zFY[i, j, k], zFZ[i, j, k]]]
                jac_det[i, j, k] =  np.linalg.det(jac_mij)

    # 3D glow
    #if nb_dims == 3:
    #    dx = J[0]
    #    dy = J[1]
    #    dz = J[2]

        # compute jacobian components
    #    Jdet0 = dx[0, ...] * (dy[1, ...] * dz[2, ...] - dy[2, ...] * dz[1, ...])
    #    Jdet1 = dx[1, ...] * (dy[0, ...] * dz[2, ...] - dy[2, ...] * dz[0, ...])
    #    Jdet2 = dx[2, ...] * (dy[0, ...] * dz[1, ...] - dy[1, ...] * dz[0, ...])

    #    return Jdet0 - Jdet1 + Jdet2

    #else:  # must be 2

    #    dfdx = J[0]
    #    dfdy = J[1]

    #    return dfdx[0, ...] * dfdy[1, ...] - dfdy[0, ...] * dfdx[1, ...]
    return jac_det

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
    disp = disp.transpose(1, 2, 3, 0)
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

def dice(y_pred, y_true, ):
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2.*intersection) / (union + 1e-5)
    return dsc

def smooth_seg(binary_img, sigma=1.5, thresh=0.5):
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img

def checkboard(shape, block_sz = 20):
    sz = 20
    xvalue = 64
    yvalue = 64
    A = np.zeros((sz, sz))
    B = np.ones((sz, sz))
    C = np.zeros((sz*xvalue, sz*yvalue))
    m = sz
    n = 0
    num = 2
    for i in range(xvalue):
        n1=0
        m1=sz
        for j in range(yvalue):
            if num % 2 == 0:
                C[n:m, n1:m1] = A
                num += 1
            else:
                C[n:m, n1:m1] = B
                num += 1
            n1 = n1 + sz
            m1 = m1 + sz
        if yvalue%2 == 0:
            num = num + 1
        n = n + sz
        m = m + sz

    C = C[0:shape[0], 0:shape[1]]
    return C