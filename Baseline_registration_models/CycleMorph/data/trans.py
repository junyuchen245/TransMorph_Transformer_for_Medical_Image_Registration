# import math
import random
import collections
import numpy as np
import torch, sys, random, math
from scipy import ndimage

from .rand import Constant, Uniform, Gaussian
from scipy.ndimage import rotate
from skimage.transform import rescale, resize

class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False): # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            # how to know  if the last dim is channel??
            # nhwtc vs nhwt??
            shape = im.shape[1:dim+1]
            # print(dim,shape) # 3, (240,240,155)
            self.sample(*shape)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)] # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

Identity = Base

# gemetric transformations, need a buffers
# first axis is N
class Rot90(Base):
    def __init__(self, axes=(0, 1)):
        self.axes = axes

        for a in self.axes:
            assert a > 0

    def sample(self, *shape):
        shape = list(shape)
        i, j = self.axes

        # shape: no first dim
        i, j = i-1, j-1
        shape[i], shape[j] = shape[j], shape[i]

        return shape

    def tf(self, img, k=0):
        return np.rot90(img, axes=self.axes)

    def __str__(self):
        return 'Rot90(axes=({}, {})'.format(*self.axes)

# class RandomRotion(Base):
#     def __init__(self, angle=20):# angle :in degress, float, [0,360]
#         assert angle >= 0.0
#         self.axes = (0,1) # 只对HW方向进行旋转
#         self.angle = angle #
#         self.buffer = None
#
#     def sample(self, *shape):# shape : [H,W,D]
#         shape = list(shape)
#         self.buffer = round(np.random.uniform(low=-self.angle,high=self.angle),2) # 2个小数点
#         if self.buffer < 0:
#             self.buffer += 180
#         return shape
#
#     def tf(self, img, k=0): # img shape [1,H,W,D,c] while label shape is [1,H,W,D]
#         return ndimage.rotate(img, angle=self.buffer, reshape=False)
#
#     def __str__(self):
#         return 'RandomRotion(axes=({}, {}),Angle:{}'.format(*self.axes,self.buffer)

class RandomRotion(Base):
    def __init__(self,angle_spectrum=10):
        assert isinstance(angle_spectrum,int)
        # axes = [(2, 1), (3, 1),(3, 2)]
        axes = [(1, 0), (2, 1),(2, 0)]
        self.angle_spectrum = angle_spectrum
        self.axes = axes

    def sample(self,*shape):
        self.axes_buffer = self.axes[np.random.choice(list(range(len(self.axes))))] # choose the random direction
        self.angle_buffer = np.random.randint(-self.angle_spectrum, self.angle_spectrum) # choose the random direction
        return list(shape)

    def tf(self, img, k=0):
        """ Introduction: The rotation function supports the shape [H,W,D,C] or shape [H,W,D]
        :param img: if x, shape is [1,H,W,D,c]; if label, shape is [1,H,W,D]
        :param k: if x, k=0; if label, k=1
        """
        bsize = img.shape[0]

        for bs in range(bsize):
            if k == 0:
                # [[H,W,D], ...]
                # print(img.shape) # (1, 128, 128, 128, 4)
                channels = [rotate(img[bs,:,:,:,c], self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0, mode='constant', cval=-1) for c in
                            range(img.shape[4])]
                img[bs,...] = np.stack(channels, axis=-1)

            if k == 1:
                img[bs,...] = rotate(img[bs,...], self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0, mode='constant', cval=-1)

        return img

    def __str__(self):
        return 'RandomRotion(axes={},Angle:{}'.format(self.axes_buffer,self.angle_buffer)


class Flip(Base):
    def __init__(self, axis=0):
        self.axis = axis

    def tf(self, img, k=0):
        return np.flip(img, self.axis)

    def __str__(self):
        return 'Flip(axis={})'.format(self.axis)

class RandomFlip(Base):
    # mirror flip across all x,y,z
    def __init__(self,axis=0):
        # assert axis == (1,2,3) # For both data and label, it has to specify the axis.
        self.axis = (1,2,3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True,False])
        self.y_buffer = np.random.choice([True,False])
        self.z_buffer = np.random.choice([True,False])
        return list(shape) # the shape is not changed

    def tf(self,img,k=0): # img shape is (1, 240, 240, 155, 4)
        if self.x_buffer:
            img = np.flip(img,axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img,axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img,axis=self.axis[2])
        return img


class RandSelect(Base):
    def __init__(self, prob=0.5, tf=None):
        self.prob = prob
        self.ops  = tf if isinstance(tf, collections.Sequence) else (tf, )
        self.buff = False

    def sample(self, *shape):
        self.buff = random.random() < self.prob

        if self.buff:
            for op in self.ops:
                shape = op.sample(*shape)

        return shape

    def tf(self, img, k=0):
        if self.buff:
            for op in self.ops:
                img = op.tf(img, k)
        return img

    def __str__(self):
        if len(self.ops) == 1:
            ops = str(self.ops[0])
        else:
            ops = '[{}]'.format(', '.join([str(op) for op in self.ops]))
        return 'RandSelect({}, {})'.format(self.prob, ops)


class CenterCrop(Base):
    def __init__(self, size):
        self.size = size
        self.buffer = None

    def sample(self, *shape):
        size = self.size
        start = [(s -size)//2 for s in shape]
        self.buffer = [slice(None)] + [slice(s, s+size) for s in start]
        return [size] * len(shape)

    def tf(self, img, k=0):
        # print(img.shape)#(1, 240, 240, 155, 4)
        return img[tuple(self.buffer)]
        # return img[self.buffer]

    def __str__(self):
        return 'CenterCrop({})'.format(self.size)

class CenterCropBySize(CenterCrop):
    def sample(self, *shape):
        assert len(self.size) == 3  # random crop [H,W,T] from img [240,240,155]
        if not isinstance(self.size, list):
            size = list(self.size)
        else:
            size = self.size
        start = [(s-i)//2 for i, s in zip(size, shape)]
        self.buffer = [slice(None)] + [slice(s, s+i) for i, s in zip(size, start)]
        return size

    def __str__(self):
        return 'CenterCropBySize({})'.format(self.size)

class RandCrop(CenterCrop):
    def sample(self, *shape):
        size = self.size
        start = [random.randint(0, s-size) for s in shape]
        self.buffer = [slice(None)] + [slice(s, s+size) for s in start]
        return [size]*len(shape)

    def __str__(self):
        return 'RandCrop({})'.format(self.size)


class RandCrop3D(CenterCrop):
    def sample(self, *shape): # shape : [240,240,155]
        assert len(self.size)==3 # random crop [H,W,T] from img [240,240,155]
        if not isinstance(self.size,list):
            size = list(self.size)
        else:
            size = self.size
        start = [random.randint(0, s-i) for i,s in zip(size,shape)]
        self.buffer = [slice(None)] + [slice(s, s+k) for s,k in zip(start,size)]
        return size

    def __str__(self):
        return 'RandCrop({})'.format(self.size)

# for data only
class RandomIntensityChange(Base):
    def __init__(self,factor):
        shift,scale = factor
        assert (shift >0) and (scale >0)
        self.shift = shift
        self.scale = scale

    def tf(self,img,k=0):
        if k==1:
            return img

        shift_factor = np.random.uniform(-self.shift,self.shift,size=[1,img.shape[1],1,1,img.shape[4]]) # [-0.1,+0.1]
        scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale,size=[1,img.shape[1],1,1,img.shape[4]]) # [0.9,1.1)
        # shift_factor = np.random.uniform(-self.shift,self.shift,size=[1,1,1,img.shape[3],img.shape[4]]) # [-0.1,+0.1]
        # scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale,size=[1,1,1,img.shape[3],img.shape[4]]) # [0.9,1.1)
        return img * scale_factor + shift_factor

    def __str__(self):
        return 'random intensity shift per channels on the input image, including'

class RandomGammaCorrection(Base):
    def __init__(self,factor):
        lower, upper = factor
        assert (lower >0) and (upper >0)
        self.lower = lower
        self.upper = upper

    def tf(self,img,k=0):
        if k==1:
            return img
        img = img + np.min(img)
        img_max = np.max(img)
        img = img/img_max
        factor = random.choice(np.arange(self.lower, self.upper, 0.1))
        gamma = random.choice([1, factor])
        if gamma == 1:
            return img
        img = img ** gamma * img_max
        img = (img - img.mean())/img.std()
        return img

    def __str__(self):
        return 'random intensity shift per channels on the input image, including'

class MinMax_norm(Base):
    def __init__(self, ):
        a = None

    def tf(self, img, k=0):
        if k == 1:
            return img
        img = (img - img.min()) / (img.max()-img.min())
        return img

class Seg_norm(Base):
    def __init__(self, ):
        a = None
        self.seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255])
    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out

class Resize_img(Base):
    def __init__(self, shape):
        self.shape = shape

    def tf(self, img, k=0):
        if k == 1:
            img = resize(img, (img.shape[0], self.shape[0], self.shape[1], self.shape[2]),
                         anti_aliasing=False, order=0)
        else:
            img = resize(img, (img.shape[0], self.shape[0], self.shape[1], self.shape[2]),
                         anti_aliasing=False, order=3)
        return img

class Pad(Base):
    def __init__(self, pad): # [0,0,0,5,0]
        self.pad = pad
        self.px = tuple(zip([0]*len(pad), pad))

    def sample(self, *shape):

        shape = list(shape)

        # shape: no first dim
        for i in range(len(shape)):
            shape[i] += self.pad[i+1]

        return shape

    def tf(self, img, k=0):
        #nhwtc, nhwt
        dim = len(img.shape)
        return np.pad(img, self.px[:dim], mode='constant')

    def __str__(self):
        return 'Pad(({}, {}, {}))'.format(*self.pad)

class Pad3DIfNeeded(Base):
    def __init__(self, shape, value=0, mask_value=0): # [0,0,0,5,0]
        self.shape = shape
        self.value = value
        self.mask_value = mask_value

    def tf(self, img, k=0):
        pad = [(0,0)]
        if k==0:
            img_shape = img.shape[1:-1]
        else:
            img_shape = img.shape[1:]
        for i, t in zip(img_shape, self.shape):
            if i < t:
                diff = t-i
                pad.append((math.ceil(diff/2),math.floor(diff/2)))
            else:
                pad.append((0,0))
        if k == 0:
            pad.append((0,0))
        pad = tuple(pad)
        if k==0:
            return np.pad(img, pad, mode='constant', constant_values=img.min())
        else:
            return np.pad(img, pad, mode='constant', constant_values=self.mask_value)

    def __str__(self):
        return 'Pad(({}, {}, {}))'.format(*self.pad)

class Noise(Base):
    def __init__(self, dim, sigma=0.1, channel=True, num=-1):
        self.dim = dim
        self.sigma = sigma
        self.channel = channel
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        if self.channel:
            #nhwtc, hwtc, hwt
            shape = [1] if len(img.shape) < self.dim+2 else [img.shape[-1]]
        else:
            shape = img.shape
        return img * np.exp(self.sigma * torch.randn(shape, dtype=torch.float32).numpy())

    def __str__(self):
        return 'Noise()'


# dim could come from shape
class GaussianBlur(Base):
    def __init__(self, dim, sigma=Constant(1.5), app=-1):
        # 1.5 pixel
        self.dim = dim
        self.sigma = sigma
        self.eps   = 0.001
        self.app = app

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        # image is nhwtc
        for n in range(img.shape[0]):
            sig = self.sigma.sample()
            # sample each channel saperately to avoid correlations
            if sig > self.eps:
                if len(img.shape) == self.dim+2:
                    C = img.shape[-1]
                    for c in range(C):
                        img[n,..., c] = ndimage.gaussian_filter(img[n, ..., c], sig)
                elif len(img.shape) == self.dim+1:
                    img[n] = ndimage.gaussian_filter(img[n], sig)
                else:
                    raise ValueError('image shape is not supported')

        return img

    def __str__(self):
        return 'GaussianBlur()'


class ToNumpy(Base):
    def __init__(self, num=-1):
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        return img.numpy()

    def __str__(self):
        return 'ToNumpy()'


class ToTensor(Base):
    def __init__(self, num=-1):
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        return torch.from_numpy(img)

    def __str__(self):
        return 'ToTensor'


class TensorType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('torch.float32', 'torch.int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.type(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'TensorType(({}))'.format(s)


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)


class Normalize(Base):
    def __init__(self, mean=0.0, std=1.0, num=-1):
        self.mean = mean
        self.std = std
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        img -= self.mean
        img /= self.std
        return img

    def __str__(self):
        return 'Normalize()'


class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            shape = op.sample(*shape)

    def tf(self, img, k=0):
        #is_tensor = isinstance(img, torch.Tensor)
        #if is_tensor:
        #    img = img.numpy()

        for op in self.ops:
            # print(op,img.shape,k)
            img = op.tf(img, k) # do not use op(img) here

        #if is_tensor:
        #    img = np.ascontiguousarray(img)
        #    img = torch.from_numpy(img)

        return img

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)