import random
import pickle
import numpy as np
import torch

M = 2 ** 32 - 1


def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)


def add_mask(x, mask, dim=1):
    mask = mask.unsqueeze(dim)
    shape = list(x.shape);
    shape[dim] += 21
    new_x = x.new(*shape).zero_()
    new_x = new_x.scatter_(dim, mask, 1.0)
    s = [slice(None)] * len(shape)
    s[dim] = slice(21, None)
    new_x[s] = x
    return new_x


def sample(x, size):
    # https://gist.github.com/yoavram/4134617
    i = random.sample(range(x.shape[0]), size)
    return torch.tensor(x[i], dtype=torch.int16)
    # x = np.random.permutation(x)
    # return torch.tensor(x[:size])


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


_shape = (240, 240, 155)


def get_all_coords(stride):
    return torch.tensor(
        np.stack([v.reshape(-1) for v in
                  np.meshgrid(
                      *[stride // 2 + np.arange(0, s, stride) for s in _shape],
                      indexing='ij')],
                 -1), dtype=torch.int16)


_zero = torch.tensor([0])


def gen_feats():
    x, y, z = 240, 240, 155
    feats = np.stack(
        np.meshgrid(
            np.arange(x), np.arange(y), np.arange(z),
            indexing='ij'), -1).astype('float32')
    shape = np.array([x, y, z])
    feats -= shape / 2.0
    feats /= shape

    return feats