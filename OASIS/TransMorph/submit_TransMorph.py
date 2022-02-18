import glob, sys
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from scipy.ndimage.interpolation import map_coordinates, zoom

def main():
    test_dir = 'D:/DATA/OASIS/Test/'
    save_dir = 'D:/DATA/OASIS/Submit/submission/task_03/'
    model_idx = -1
    weights = [1, 1, 1]
    model_folder = 'TransMorphLarge_ncc_{}_dsc_{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    model_dir = 'experiments/' + model_folder
    config = CONFIGS_TM['TransMorph-Large']
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    file_names = glob.glob(test_dir + '*.pkl')
    with torch.no_grad():
        stdy_idx = 0
        for data in file_names:
            x, y, x_seg, y_seg = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            file_name = file_names[stdy_idx].split('\\')[-1].split('.')[0][2:]
            print(file_name)
            model.eval()
            x_in = torch.cat((x, y),dim=1)
            x_def, flow = model(x_in)
            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            np.savez(save_dir+'disp_{}.npz'.format(file_name), flow)
            stdy_idx += 1

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