'''
Probabilistic TransMorph with Diffeomorphism

Paper:
Chen, J., Du, Y., He, Y., Segars, P. W., Li, Y., & Frey, E. C. (2021). 
TransMorph: Transformer for Unsupervised Medical Image Registration. arXiv preprint.

Base code for probabilistic registration was obtained from: https://github.com/uncbiag/easyreg

Original paper for learning-based probabilistic registration:
Dalca, A. V., Balakrishnan, G., Guttag, J., & Sabuncu, M. R. (2019).
Unsupervised learning of probabilistic diffeomorphic registration for images and surfaces.
Medical image analysis, 57, 226-236.

Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import models.finite_differences as fdt
import models.TransMorph as TM
import models.configs_TransMorph_diff as configs

dim = 3
reproduce_paper_result = False

def not_normalized_identity_map(sz):
    """
    Returns an identity map.
    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0: sz[0]]
    elif dim == 2:
        id = np.mgrid[0: sz[0], 0: sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[0: sz[0], 0:sz[1], 0: sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    # id= id*2-1
    return torch.from_numpy(id.astype(np.float32))

def identity_map_for_reproduce(sz):
    """
    Returns an identity map.
    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[-1:1.:2. / sz[0]]
    elif dim == 2:
        id = np.mgrid[-1.:1.:2. / sz[0], -1.:1.:2. / sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[-1.:1.:2. / sz[0], -1.:1.:2. / sz[1], -1.:1.:2. / sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    # id= id*2-1
    return torch.from_numpy(id.astype(np.float32))

def identity_map(sz, dtype= np.float32):
    """
    Returns an identity map.
    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0: sz[0]]
    elif dim == 2:
        id = np.mgrid[0: sz[0], 0: sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[0: sz[0], 0:sz[1], 0: sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index
    spacing = 1./ (np.array(sz)-1)

    for d in range(dim):
        id[d] *= spacing[d]
        id[d] = id[d]*2 - 1

    return torch.from_numpy(id.astype(np.float32))

def gen_identity_map(img_sz, resize_factor=1.,normalized=True):
    """
    given displacement field,  add displacement on grid field
    """
    if isinstance(resize_factor, list):
        img_sz = [int(img_sz[i] * resize_factor[i]) for i in range(dim)]
    else:
        img_sz = [int(img_sz[i] * resize_factor) for i in range(dim)]
    if normalized:
        grid = identity_map(img_sz) if not reproduce_paper_result else identity_map_for_reproduce(img_sz)
    else:
        grid = not_normalized_identity_map(img_sz)
    return grid

def scale_map(map,spacing):
    """
    Scales the map to the [-1,1]^d format
    :param map: map in BxCxXxYxZ format
    :param spacing: spacing in XxYxZ format
    :return: returns the scaled map
    """
    sz = map.size()
    map_scaled = torch.zeros_like(map)
    ndim = len(spacing)

    # This is to compensate to get back to the [-1,1] mapping of the following form
    # id[d]*=2./(sz[d]-1)
    # id[d]-=1.

    for d in range(ndim):
        if sz[d+2] >1:
            map_scaled[:, d, ...] = map[:, d, ...] * (2. / (sz[d + 2] - 1.) / spacing[d]) - 1.
        else:
            map_scaled[:, d, ...] = map[:,d,...]

    return map_scaled

def scale_map_grad(grad_map,spacing):
    """
    Scales the gradient back
    :param grad_map: gradient (computed based on map normalized to [-1,1]
    :param spacing: spacing in XxYxZ format
    :return: n/a (overwrites grad_map; results in gradient based on original spacing)
    """

    # need to compensate for the rescaling of the gradient in the backward direction
    sz = grad_map.size()
    ndim = len(spacing)
    for d in range(ndim):
        #grad_map[:, d, ...] *= spacing[d] * (sz[d + 2] - 1) / 2.
        #grad_map[:, d, ...] *= (sz[d + 2] - 1)/2.
        if sz[d + 2] > 1:
            grad_map[:, d, ...]  *= (2. / (sz[d + 2] - 1.) / spacing[d])
        else:
            grad_map[:, d, ...] = grad_map[:, d, ...]

class STNFunction_ND_BCXYZ(nn.Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, spacing, zero_boundary = False,using_bilinear=True,using_01_input=True):
        """
        Constructor
        :param ndim: (int) spatial transformation of the transform
        """
        super(STNFunction_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        self.ndim = len(spacing)
        #zero_boundary = False
        self.zero_boundary = 'zeros' if zero_boundary else 'border'
        self.mode = 'bilinear' if using_bilinear else 'nearest'
        self.using_01_input=using_01_input

    def forward_stn(self, input1, input2, ndim):
        if ndim==1:
            # use 2D interpolation to mimick 1D interpolation
            # now test this for 1D
            phi_rs = input2.reshape(list(input2.size()) + [1])
            input1_rs = input1.reshape(list(input1.size()) + [1])

            phi_rs_size = list(phi_rs.size())
            phi_rs_size[1] = 2

            phi_rs_ordered = torch.zeros(phi_rs_size,dtype=phi_rs.dtype,device=phi_rs.device)
            # keep dimension 1 at zero
            phi_rs_ordered[:, 1, ...] = phi_rs[:, 0, ...]

            output_rs = torch.nn.functional.grid_sample(input1_rs, phi_rs_ordered.permute([0, 2, 3, 1]), mode=self.mode, padding_mode=self.zero_boundary,align_corners=True)
            output = output_rs[:, :, :, 0]

        if ndim==2:
            # todo double check, it seems no transpose is need for 2d, already in height width design
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:,0,...] = input2[:,1,...]
            input2_ordered[:,1,...] = input2[:,0,...]
            output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 1]), mode=self.mode,
                                          padding_mode=self.zero_boundary,align_corners=True)
        if ndim==3:
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:, 0, ...] = input2[:, 2, ...]
            input2_ordered[:, 1, ...] = input2[:, 1, ...]
            input2_ordered[:, 2, ...] = input2[:, 0, ...]
            output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]), mode=self.mode, padding_mode=self.zero_boundary,align_corners=True)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform
        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """

        assert(len(self.spacing)+2==len(input2.size()))
        if self.using_01_input:
            output = self.forward_stn(input1, scale_map(input2,self.spacing), self.ndim)
        else:
            output = self.forward_stn(input1, input2, self.ndim)
        # print(STNVal(output, ini=-1).sum())
        return output

class STN_ND_BCXYZ(nn.Module):
    """
    Spatial transform code for nD spatial transoforms. Uses the BCXYZ image format.
    """
    def __init__(self, spacing, zero_boundary=False,use_bilinear=True,use_01_input=True,):
        super(STN_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        """spatial dimension"""
        self.f = STNFunction_ND_BCXYZ( self.spacing,zero_boundary= zero_boundary,using_bilinear= use_bilinear,using_01_input = use_01_input)

        """spatial transform function"""
    def forward(self, input1, input2):
        """
       Simply returns the transformed input
       :param input1: image in BCXYZ format
       :param input2: map in BdimXYZ format
       :return: returns the transformed image
       """
        return self.f(input1, input2)

class convBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, batchnorm=False, residual=False, nonlinear=nn.LeakyReLU(0.2)):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(convBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinear = nonlinear
        self.residual = residual

    def forward(self, x):
        x= self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.nonlinear:
            x = self.nonlinear(x)
        if self.residual:
            x += x

        return x

class Bilinear(nn.Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, zero_boundary=False, using_scale=False, mode='bilinear'):
        """
        Constructor
        :param ndim: (int) spatial transformation of the transform
        """
        super(Bilinear, self).__init__()
        self.zero_boundary = 'zeros' if zero_boundary else 'border'
        self.using_scale = using_scale
        """ scale [-1,1] image intensity into [0,1], this is due to the zero boundary condition we may use here """
        self.mode = mode

    def forward_stn(self, input1, input2):
        input2_ordered = torch.zeros_like(input2)
        input2_ordered[:, 0, ...] = input2[:, 2, ...]
        input2_ordered[:, 1, ...] = input2[:, 1, ...]
        input2_ordered[:, 2, ...] = input2[:, 0, ...]

        output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]), mode=self.mode,
                                                     padding_mode=self.zero_boundary, align_corners=True)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform
        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
        if self.using_scale:

            output = self.forward_stn((input1 + 1) / 2, input2)
            # print(STNVal(output, ini=-1).sum())
            return output * 2 - 1
        else:
            output = self.forward_stn(input1, input2)
            # print(STNVal(output, ini=-1).sum())
            return output


class TransMorphDiff(nn.Module):
    """
    Probabilistic TransMorph Model
    :return: Warped image, Deformation field, Displacement field
    """
    def __init__(self, config):
        super(TransMorphDiff, self).__init__()
        img_sz = config.img_size
        image_sigma = config.image_sigma
        prior_lambda = config.prior_lambda
        prior_lambda_mean = config.prior_lambda_mean
        self.input_channel = config.in_chans
        self.output_channel = 3
        self.img_sz = img_sz
        self.low_res_img_sz = [int(x/2) for x in img_sz]
        self.spacing = 1. / ( np.array(img_sz) - 1)
        self.int_steps = 7
        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.prior_lambda_mean = prior_lambda_mean
        self.flow_vol_shape = self.low_res_img_sz
        self.D = self._degree_matrix(self.flow_vol_shape)
        self.D = (self.D).cuda()
        self.loss_fn =  None

        self.id_transform = gen_identity_map(self.img_sz, 1.0).cuda()
        self.id_transform  =self.id_transform.view([1]+list(self.id_transform.shape))

        """to compatiable to the mesh setting in voxel morph"""
        self.low_res_id_transform = gen_identity_map(self.img_sz, 0.5, normalized=False).cuda()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bilinear = STN_ND_BCXYZ(np.array([1.,1.,1.]),zero_boundary=True)
        self.bilinear_img = Bilinear(zero_boundary=True)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = TM.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf)
        self.up0 = TM.DecoderBlock(embed_dim * 8, embed_dim * 4, skip_channels=embed_dim * 4 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = TM.DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)
        self.up2 = TM.DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)
        self.up3 = TM.DecoderBlock(embed_dim, config.reg_head_chan, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)
        self.c1 = TM.Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)

        self.flow_mean =  nn.Conv3d(config.reg_head_chan, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.flow_sigma =  nn.Conv3d(config.reg_head_chan, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.flow_mean.weight.data.normal_(0.,1e-5)
        self.flow_sigma.weight.data.normal_(0.,1e-10)
        self.flow_sigma.bias.data = torch.Tensor([-10]*3)
        self.print_count=0
        # identity transform for computing displacement

    def scale_map(self,map, spacing):
        """
        Scales the map to the [-1,1]^d format
        :param map: map in BxCxXxYxZ format
        :param spacing: spacing in XxYxZ format
        :return: returns the scaled map
        """
        sz = map.size()
        map_scaled = torch.zeros_like(map)
        ndim = len(spacing)

        # This is to compensate to get back to the [-1,1] mapping of the following form
        # id[d]*=2./(sz[d]-1)
        # id[d]-=1.

        for d in range(ndim):
            if sz[d + 2] > 1:
                map_scaled[:, d, ...] = map[:, d, ...] * (2. / (sz[d + 2] - 1.) / spacing[d])
            else:
                map_scaled[:, d, ...] = map[:, d, ...]

        return map_scaled

    def set_loss_fn(self, loss_fn):
        """ set loss function"""
        pass

    def forward(self, inputs):
        source, target = inputs
        self.__do_some_clean()
        affine_map = self.id_transform.clone()
        x = torch.cat((source, target), dim=1)
        if self.if_convskip:
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None
        out = self.transformer(x)

        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        flow_mean = self.flow_mean(x)
        log_sigma = self.flow_sigma(x)
        noise = torch.randn(flow_mean.shape).cuda()
        if self.training:
            flow = flow_mean + torch.exp(log_sigma / 2.0) * noise
        else:
            flow = flow_mean
        for _ in range(self.int_steps):
            deform_field = flow + self.low_res_id_transform
            flow_1 = self.bilinear(flow, deform_field)
            flow = flow_1 + flow
        disp_field = F.interpolate(flow, scale_factor=2, mode='trilinear')
        disp_field = self.scale_map(disp_field,np.array([1,1,1]))
        deform_field = disp_field + affine_map
        warped_source = self.bilinear_img(source, deform_field)
        self.res_flow_mean  = flow
        self.res_log_sigma = log_sigma
        self.warped = warped_source
        self.target = target
        self.source = source

        return warped_source, deform_field, disp_field

    def check_if_update_lr(self):
        return False, None

    def get_extra_to_plot(self):
        return None, None
    def __do_some_clean(self):
        self.res_flow_mean = None
        self.res_log_sigma = None
        self.warped = None
        self.target = None
        self.source = None

    def scale_reg_loss(self,):
        reg = self.kl_loss()
        return reg

    def get_sim_loss(self,):
        loss = self.recon_loss()
        return loss

    def get_id_loss(self,):
        loss = torch.mean((self.flow_id)**2)
        return loss

    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently,
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)  # 3 3 3
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied
        # ith feature to ith feature
        filt = np.zeros([ndims, ndims] + [3] * ndims)  # 3 3 3 3  ##!!!!!!!! in out w h d
        for i in range(ndims):
            filt[i, i, ...] = filt_inner  ##!!!!!!!!

        return filt

    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [ndims,*vol_shape]  # 96 96 40 3  ##!!!!!!!!

        # prepare conv kernel
        conv_fn = F.conv3d  ##!!!!!!!!

        # prepare tf filter
        z = torch.ones([1] + sz)  # 1 96 96 40 3
        filt_tf = torch.Tensor(self._adj_filt(ndims))  # 3 3 3 3 ##!!!!!!!!
        strides = [1] * (ndims)  ##!!!!!!!!
        return conv_fn(z, filt_tf, padding= 1, stride =strides)  ##!!!!!!!!

    def prec_loss(self, disp):  ##!!!!!!!!
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i
        Note: could probably do with a difference filter,
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        fd = fdt.FD_torch(np.array([1., 1., 1.]))
        dfx = fd.dXc(disp[:, 0, ...])
        dfy = fd.dYc(disp[:, 1, ...])
        dfz = fd.dZc(disp[:, 2, ...])
        l2 = dfx ** 2 + dfy ** 2 + dfz ** 2
        reg = l2.mean()
        return reg * 0.5

    def kl_loss(self):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3
        y_true is only used to get the shape
        """
        # prepare inputs
        ndims = 3
        flow_mean = self.res_flow_mean
        log_sigma = self.res_log_sigma

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims,
        # which is a function of the data

        # sigma terms
        sigma_term = self.prior_lambda * self.D * torch.exp(log_sigma) - log_sigma  ##!!!!!!!!
        sigma_term = torch.mean(sigma_term)  ##!!!!!!!!

        # precision terms
        prec_term = self.prior_lambda_mean * self.prec_loss(flow_mean)  # this is the jacobi loss

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well

    def recon_loss(self):
        """ reconstruction loss """
        y_pred = self.warped
        y_true = self.target
        return 1. / (self.image_sigma ** 2) * torch.mean((y_true - y_pred)**2)  ##!!!!!!!!

    def get_loss(self):
        sim_loss = self.get_sim_loss()
        reg_loss = self.scale_reg_loss()
        return sim_loss+ reg_loss

    def get_inverse_map(self,):
        _, inverse_map = self.forward(self.target, self.source)
        return inverse_map

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()


CONFIGS = {
    'TransMorphDiff': configs.get_TransMorphDiff_config(),
}
