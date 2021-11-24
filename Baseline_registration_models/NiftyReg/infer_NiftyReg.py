import glob
import os, utils, torch
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
from torchvision import transforms
import nibabel as nib

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def main():
    test_dir = '/mnt/d/DATA/JHUBrain/Test/'
    dict = utils.process_label()
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line, 'NiftyReg')
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
            x_nib = nib.Nifti1Image(x, np.eye(4))
            x_nib.header.get_xyzt_units()
            x_nib.to_filename('x.nii.gz')

            y_nib = nib.Nifti1Image(y, np.eye(4))
            y_nib.header.get_xyzt_units()
            y_nib.to_filename('y.nii.gz')

            xseg_nib = nib.Nifti1Image(x_seg, np.eye(4))
            xseg_nib.header.get_xyzt_units()
            xseg_nib.to_filename('xseg.nii.gz')

            yseg_nib = nib.Nifti1Image(y_seg, np.eye(4))
            yseg_nib.header.get_xyzt_units()
            yseg_nib.to_filename('yseg.nii.gz')

            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_f3d -be 0.0002 --ssd -ref y.nii.gz -flo x.nii.gz -res output_deformed.nii.gz -cpp ref_template_flo_new_image_nrr_cpp.nii')
            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_resample -ref y.nii.gz -flo xseg.nii.gz -res output_deformed_seg.nii.gz -cpp ref_template_flo_new_image_nrr_cpp.nii -inter 0')
            def_seg = nib.load('output_deformed_seg.nii.gz')
            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_transform -ref y.nii.gz -cpp2def ref_template_flo_new_image_nrr_cpp.nii def.nii.gz')
            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_transform -ref y.nii.gz -def2disp def.nii disp.nii.gz')
            flow = nib.load('disp.nii.gz')
            flow = flow.get_fdata()
            flow = flow[..., 0, :].transpose(3, 0, 1, 2)
            def_seg = def_seg.get_fdata()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            tar_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_trans = utils.dice_val(def_seg.long(), tar_seg.long(), 46)
            line = utils.dice_val_substruct(def_seg.long(), tar_seg.long(), stdy_idx)
            jac_det = utils.jacobian_determinant_vxm(flow)
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(y_seg.shape), 1)
            csv_writter(line, 'NiftyReg')
            eval_dsc_def.update(dsc_trans.item(), 1)
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            stdy_idx += 1

            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_f3d -be 0.0002 -ssd -ref x.nii.gz -flo y.nii.gz -res output_deformed.nii.gz -cpp ref_template_flo_new_image_nrr_cpp.nii')
            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_resample -ref x.nii.gz -flo yseg.nii.gz -res output_deformed_seg.nii.gz -cpp ref_template_flo_new_image_nrr_cpp.nii -inter 0')
            def_seg = nib.load('output_deformed_seg.nii.gz')
            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_transform -ref x.nii.gz -cpp2def ref_template_flo_new_image_nrr_cpp.nii def.nii.gz')
            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_transform -ref x.nii.gz -def2disp def.nii disp.nii.gz')
            flow = nib.load('disp.nii.gz')
            flow = flow.get_fdata()
            flow = flow[..., 0, :].transpose(3, 0, 1, 2)
            def_seg = def_seg.get_fdata()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            tar_seg = torch.from_numpy(x_seg[None, None, ...])
            dsc_trans = utils.dice_val(def_seg.long(), tar_seg.long(), 46)
            line = utils.dice_val_substruct(def_seg.long(), tar_seg.long(), stdy_idx)
            jac_det = utils.jacobian_determinant_vxm(flow)
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(y_seg.shape), 1)
            csv_writter(line, 'NiftyReg')
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