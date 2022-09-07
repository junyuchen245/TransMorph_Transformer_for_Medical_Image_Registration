import glob
import os, utils, torch
from torch.utils.data import DataLoader
from data_IXI import datasets, trans
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import nibabel as nib
import torch.nn as nn

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def main():
    atlas_dir = 'Path_to_IXI_data/atlas.pkl'
    test_dir = 'Path_to_IXI_data/Val/'
    dict = utils.process_label()
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line + ',' + 'non_jec', 'NiftyReg_IXI')
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    stdy_idx = 0
    eval_dsc_def = AverageMeter()
    eval_det = AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
            y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
            x_seg = data[2]
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            x_seg_oh = x_seg_oh.squeeze(0).detach().cpu().numpy()

            y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()

            x_nib = nib.Nifti1Image(x, np.eye(4))
            x_nib.header.get_xyzt_units()
            x_nib.to_filename('x.nii.gz')

            y_nib = nib.Nifti1Image(y, np.eye(4))
            y_nib.header.get_xyzt_units()
            y_nib.to_filename('y.nii.gz')

            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_f3d -be 0.0006 --ssd -ref y.nii.gz -flo x.nii.gz -res output_deformed.nii.gz -cpp ref_template_flo_new_image_nrr_cpp.nii')
            def_segs = []
            for i in range(x_seg_oh.shape[0]):
                xchan_nib = nib.Nifti1Image(x_seg_oh[i].astype(np.float32), np.eye(4))
                xchan_nib.header.get_xyzt_units()
                xchan_nib.to_filename('xseg.nii.gz')
                os.system(
                    '/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_resample -ref y.nii.gz -flo xseg.nii.gz -res output_deformed_seg.nii.gz -cpp ref_template_flo_new_image_nrr_cpp.nii -LIN')
                def_seg = nib.load('output_deformed_seg.nii.gz')
                def_seg = def_seg.get_fdata()
                def_segs.append(def_seg[None, ...])
            def_segs = np.concatenate(def_segs, axis=0)
            def_seg = np.argmax(def_segs, axis=0)
            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_transform -ref y.nii.gz -cpp2def ref_template_flo_new_image_nrr_cpp.nii def.nii.gz')
            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_transform -ref y.nii.gz -def2disp def.nii disp.nii.gz')
            flow = nib.load('disp.nii.gz')
            flow = flow.get_fdata()
            flow = flow[..., 0, :].transpose(3, 0, 1, 2)
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            tar_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_trans = utils.dice_val(def_seg.long(), tar_seg.long(), 46)
            line = utils.dice_val_substruct(def_seg.long(), tar_seg.long(), stdy_idx)
            jac_det = utils.jacobian_determinant_vxm(flow)
            line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(y_seg.shape))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(y_seg.shape), 1)
            csv_writter(line, file_name)
            eval_dsc_def.update(dsc_trans.item(), 1)
            print('DSC: {:.4f}'.format(dsc_trans.item()))
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
    main()
