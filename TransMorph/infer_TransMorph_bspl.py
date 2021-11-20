import os, losses, utils, glob
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import models.transformation as transformation
from models.TransMorph_bspl import CONFIGS as CONFIGS_TM
import models.TransMorph_bspl as TransMorph_bspl

def main():
    test_dir = 'D:/DATA/JHUBrain/Test/'
    model_idx = -1
    weights = [1, 0.02]
    model_folder = 'TransMorphBSpline_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    dict = utils.process_label()
    if os.path.exists('experiments/'+model_folder[:-1]+'_jecdet.csv'):
        os.remove('experiments/'+model_folder[:-1]+'_jecdet.csv')
    csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1]+'_jecdet')
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line, 'experiments/' + model_folder[:-1])
    config = CONFIGS_TM['TransMorphBSpline-v0']
    model = TransMorph_bspl.TranMorphBSplineNet(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.JHUBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_def, flow, disp = model((x,y))
            flow = disp
            def_out = transformation.warp(x_seg.cuda().float(), disp.cuda(), interp_mode='nearest')
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

            # flip moving and fixed images
            y_def, flow, disp = model((y, x))
            flow = disp
            def_out = transformation.warp(y_seg.cuda().float(), disp.cuda(), interp_mode='nearest')
            tar = x.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
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