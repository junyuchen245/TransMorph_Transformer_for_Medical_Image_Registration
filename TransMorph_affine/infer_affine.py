import os, utils, glob, losses
import pickle
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, TransMorh_affine
from torchvision import transforms
import torch.nn as nn
from TransMorh_affine import CONFIGS as CONFIGS_TM
from natsort import natsorted

def savepkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def main():
    test_dir = 'D:/DATA/Duke/All_adult/'


    AffInfer = TransMorh_affine.ApplyAffine()
    AffInfer.cuda()
    AffInferNN = TransMorh_affine.ApplyAffine(mode='nearest')
    AffInferNN.cuda()

    config = CONFIGS_TM['TransMorph-Affine']
    AffModel = TransMorh_affine.TransMorphAffine(config)
    AffModel.load_state_dict(torch.load('experiments/TransMorph_Affine/' + natsorted(os.listdir('experiments/TransMorph_Affine/'))[0])['state_dict'])
    AffModel.cuda()

    reg_model_bilin = utils.register_model((160, 160, 160), 'bilinear')
    reg_model_bilin.cuda()
    reg_model_NN = utils.register_model((160, 160, 160), 'nearest')
    reg_model_NN.cuda()

    test_composed = transforms.Compose([trans.Pad3DIfNeeded((180, 180, 180)),
                                       trans.CenterCropBySize((160, 160, 160)),
                                       trans.NumpyType((np.float32, np.float32)),
                                       ])
    files = glob.glob(test_dir + '*.pkl')
    test_set = datasets.CTSegDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    idx = 0
    for data in test_loader:
        print(files[idx].split('\\')[-1].split('.')[0])
        file_name = files[idx].split('\\')[-1].split('.')[0]
        AffModel.eval()
        data = [t.cuda() for t in data]

        ####################
        # Affine transform
        ####################
        x = data[0]; x_seg = data[2]
        y = data[1]; y_seg = data[3]
        x_in = torch.cat((x, y), dim=1)
        ct_aff, mats, inv_mats = AffModel(x_in)
        phan = y.detach().clone(); phan_seg = y_seg
        phan_seg = nn.functional.one_hot(phan_seg.long(), num_classes=16)
        phan_seg = torch.squeeze(phan_seg, 1)
        phan_seg = phan_seg.permute(0, 4, 1, 2, 3).contiguous()
        ct_tar_seg = AffInferNN(x_seg.float(), mats.float())
        ct_aff = ct_aff.cpu().detach().numpy()[0, 0, :, :, :]
        ct_tar_seg = ct_tar_seg.cpu().detach().numpy()[0, 0, :, :, :]
        savepkl(data=(ct_aff, ct_tar_seg), path='D:/DATA/Duke/All_adult_affine/' + file_name + '.pkl')
        idx += 1

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