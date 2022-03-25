import glob
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
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def main():
    test_dir = 'E:/Junyu/DATA/RaFD/Test/'
    model_idx = -1
    weights = [1, 1]
    model_folder = 'TransMorph_ssim_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder

    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+model_folder[:-1]+'.csv'):
        os.remove('Quantitative_Results/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + model_folder[:-1])
    line = ',SSIM,det'
    csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])

    config = CONFIGS_TM['TransMorph-Sin']
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()
    test_set = datasets.RaFDInferDataset(glob.glob(test_dir + '*.pkl'), transforms=None)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    ssim = SSIM(data_range=255, size_average=True, channel=1)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x_rgb = data[0]
            y_rgb = data[1]
            x = data[2]
            y = data[3]

            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            ncc = ssim(y, x)
            eval_dsc_raw.update(ncc.item(), x.numel())
            ncc = ssim(output[0], x)
            eval_dsc_def.update(ncc.item(), x.numel())
            jac_det = utils.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(x.shape), x.numel())
            line = 'p{}'.format(stdy_idx) + ',' + str(ncc.item()) + ',' + str(np.sum(jac_det <= 0) / np.prod(x.shape))
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])
            stdy_idx += 1
            # flip image
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            ncc = ssim(y, x)
            eval_dsc_raw.update(ncc.item(), x.numel())
            ncc = ssim(output[0], y)
            eval_dsc_def.update(ncc.item(), y.numel())
            jac_det = utils.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :])
            line = 'p{}'.format(stdy_idx) + ',' + str(ncc.item()) + ',' + str(np.sum(jac_det <= 0) / np.prod(x.shape))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(x.shape), x.numel())
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])
            stdy_idx += 1
        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
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