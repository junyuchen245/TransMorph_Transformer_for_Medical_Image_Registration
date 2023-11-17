from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses, random, math
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_affine import CONFIGS as CONFIGS_TM
import models.TransMorph_affine as TransMorph
import torch.nn as nn

def affine_aug(im, im_label=None, seed=10):
    # mode = 'bilinear' or 'nearest'
    with torch.no_grad():
        random.seed(seed)
        angle_range = 10
        trans_range = 0.1
        scale_range = 0.1
        # scale_range = 0.15

        angle_xyz = (random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180)
        scale_xyz = (random.uniform(-scale_range, scale_range), random.uniform(-scale_range, scale_range),
                     random.uniform(-scale_range, scale_range))
        trans_xyz = (random.uniform(-trans_range, trans_range), random.uniform(-trans_range, trans_range),
                     random.uniform(-trans_range, trans_range))

        rotation_x = torch.tensor([
            [1., 0, 0, 0],
            [0, math.cos(angle_xyz[0]), -math.sin(angle_xyz[0]), 0],
            [0, math.sin(angle_xyz[0]), math.cos(angle_xyz[0]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_y = torch.tensor([
            [math.cos(angle_xyz[1]), 0, math.sin(angle_xyz[1]), 0],
            [0, 1., 0, 0],
            [-math.sin(angle_xyz[1]), 0, math.cos(angle_xyz[1]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_z = torch.tensor([
            [math.cos(angle_xyz[2]), -math.sin(angle_xyz[2]), 0, 0],
            [math.sin(angle_xyz[2]), math.cos(angle_xyz[2]), 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        trans_shear_xyz = torch.tensor([
            [1. + scale_xyz[0], 0, 0, trans_xyz[0]],
            [0, 1. + scale_xyz[1], 0, trans_xyz[1]],
            [0, 0, 1. + scale_xyz[2], trans_xyz[2]],
            [0, 0, 0, 1]
        ], requires_grad=False).unsqueeze(0).cuda()

        theta_final = torch.matmul(rotation_x, rotation_y)
        theta_final = torch.matmul(theta_final, rotation_z)
        theta_final = torch.matmul(theta_final, trans_shear_xyz)

        output_disp_e0_v = F.affine_grid(theta_final[:, 0:3, :], im.shape, align_corners=False)

        im = F.grid_sample(im, output_disp_e0_v, mode='bilinear', padding_mode="border", align_corners=False)

        if im_label is not None:
            im_label = F.grid_sample(im_label, output_disp_e0_v, mode='nearest', padding_mode="border",
                                     align_corners=False)
            return im, im_label
        else:
            return im

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = 1
    atlas_dir = 'D:/DATA/IXI/atlas.pkl'
    train_dir = 'D:/DATA/IXI/toy_train/'
    val_dir = 'D:/DATA/IXI/toy/'
    save_dir = 'TransMorphAffine_ncc/'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.1 # learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    H, W, D = 160, 192, 224
    config = CONFIGS_TM['TransMorph']
    config.img_size = (H, W, D)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    config.embed_dim = 96
    config.use_checkpoint = False
    model = TransMorph.TransMorphAffine(config)
    model.cuda()
    affine_trans = TransMorph.AffineTransform()#AffineTransformer((H, W, D)).cuda()
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            atlas = data[0]
            pat_img = data[1]
            x_ = affine_aug(pat_img, seed=idx)
            y_ = atlas
            aff, scl, transl, shr = model((x_, y_))
            x_trans, mat, inv_mat = affine_trans(x_, aff, scl, transl, shr)
            loss = criterion(x_trans, y_)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}'.format(idx, len(train_loader), loss.item()))
        print('Epoch {}'.format(epoch))
        '''
        Validation
        '''
        idd = 0
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                atlas = data[0]
                pat_img = data[1]
                x_ = affine_aug(pat_img, seed=idd)
                y_ = atlas  # affine_aug(y_half)
                aff, scl, transl, shr = model((x_, y_))
                x_trans, mat, inv_mat = affine_trans(x_, aff, scl, transl, shr)
                y_trans = affine_trans.apply_affine(y_, inv_mat)
                print(mat)
                plt.figure()
                plt.subplot(2, 2, 1)
                plt.imshow(x_.cpu().detach().numpy()[0, 0, :, 96, ])
                plt.subplot(2, 2, 2)
                plt.imshow(y_.cpu().detach().numpy()[0, 0, :, 96, ])
                plt.subplot(2, 2, 4)
                plt.imshow(x_trans.cpu().detach().numpy()[0, 0, :, 96, ])
                plt.subplot(2, 2, 3)
                plt.imshow(y_trans.cpu().detach().numpy()[0, 0, :, 96, ])
                plt.savefig('reg_results{}'.format(idd))
                plt.close()
                idd += 1


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()