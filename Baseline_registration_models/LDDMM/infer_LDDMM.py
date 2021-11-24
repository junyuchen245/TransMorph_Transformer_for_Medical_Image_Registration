import glob
import os, utils, torch
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
from torchvision import transforms
import nibabel as nib
import torch_lddmm

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def main():
    test_dir = 'D:/DATA/JHUBrain/Test/'
    dict = utils.process_label()
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line, 'lddmm')
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.JHUBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    stdy_idx = 0
    eval_dsc_def = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
            y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
            x_seg = data[2].squeeze(0).squeeze(0).detach().cpu().numpy()
            y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()
            dx = [1, 1, 1]
            lddmm = torch_lddmm.LDDMM(template=x * 255., target=y * 255., outdir='./', do_affine=0, do_lddmm=1, a=5.,
                                      p=2,
                                      niter=500, epsilon=2, sigma=4., sigmaR=10., optimizer='sgd', dx=dx, nt=7,
                                      gpu_number=0,
                                      minbeta=1e-10)
            lddmm.run()
            (def_seg, _, _, _) = lddmm.applyThisTransform(x_seg, interpmode='nearest')
            flow = lddmm.computeThisDisplacement()
            flow = np.stack(flow, axis=0)
            def_seg = def_seg[-1].cpu().numpy()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            tar_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_trans = utils.dice_val(def_seg.long(), tar_seg.long(), 46)
            eval_dsc_def.update(dsc_trans.item(), 1)
            jac_det = utils.jacobian_determinant_vxm(flow)
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            line = utils.dice_val_substruct(def_seg.long(), tar_seg.long(), stdy_idx)
            line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(y_seg.shape))
            csv_writter(line, 'lddmm')
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(y_seg.shape), 1)
            stdy_idx += 1

            #flip image
            lddmm = torch_lddmm.LDDMM(template=y * 255., target=x * 255., outdir='./', do_affine=0, do_lddmm=1, a=5.,
                                      p=2,
                                      niter=500, epsilon=2, sigma=4., sigmaR=10., optimizer='sgd', dx=dx, nt=7,
                                      gpu_number=0,
                                      minbeta=1e-10)
            lddmm.run()
            (def_seg, _, _, _) = lddmm.applyThisTransform(y_seg, interpmode='nearest')
            flow = lddmm.computeThisDisplacement()
            flow = np.stack(flow, axis=0)
            def_seg = def_seg[-1].cpu().numpy()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            tar_seg = torch.from_numpy(x_seg[None, None, ...])
            jac_det = utils.jacobian_determinant_vxm(flow)
            line = utils.dice_val_substruct(def_seg.long(), tar_seg.long(), stdy_idx)
            line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(y_seg.shape))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            csv_writter(line, 'lddmm')
            eval_dsc_def.update(dsc_trans.item(), 1)
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(y_seg.shape), 1)
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
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