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
    csv_writter(line, 'deedsBCV')
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

            os.system('/mnt/d/pythonProject/deedsBCV_reg/deedsBCV/deedsBCV -F y.nii.gz -M x.nii.gz -O output -S xseg.nii.gz -G 6x5x4x3x2 -L 6x5x4x3x2 -Q 5x4x3x2x1')
            def_seg = nib.load('output_deformed_seg.nii.gz')
            def_seg = def_seg.get_fdata()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            tar_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_trans = utils.dice_val(def_seg.long(), tar_seg.long(), 46)
            line = utils.dice_val_substruct(def_seg.long(), tar_seg.long(), stdy_idx)
            flow = nib.load('dense_disp.nii.gz')
            flow = flow.get_fdata().transpose(3, 0, 1, 2)
            jec_det = utils.jacobian_determinant_vxm(flow)
            line = line + ',' + str(np.sum(jec_det <= 0) / np.prod(y_seg.shape))
            print('det < 0: {}'.format(np.sum(jec_det <= 0) / np.prod(y.shape)))
            eval_det.update(np.sum(jec_det <= 0) / np.prod(y.shape), 1)
            csv_writter(line, 'deedsBCV')
            eval_dsc_def.update(dsc_trans.item(), 1)
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            stdy_idx += 1

            os.system('/mnt/d/pythonProject/deedsBCV_reg/deedsBCV/deedsBCV -F x.nii.gz -M y.nii.gz -O output -S yseg.nii.gz -G 6x5x4x3x2 -L 6x5x4x3x2 -Q 5x4x3x2x1')
            def_seg = nib.load('output_deformed_seg.nii.gz')
            def_seg = def_seg.get_fdata()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            tar_seg = torch.from_numpy(x_seg[None, None, ...])
            dsc_trans = utils.dice_val(def_seg.long(), tar_seg.long(), 46)
            line = utils.dice_val_substruct(def_seg.long(), tar_seg.long(), stdy_idx)
            flow = nib.load('dense_disp.nii.gz')
            flow = flow.get_fdata().transpose(3, 0, 1, 2)
            jec_det = utils.jacobian_determinant_vxm(flow)
            print('det < 0: {}'.format(np.sum(jec_det <= 0) / np.prod(y.shape)))
            line = line + ',' + str(np.sum(jec_det <= 0) / np.prod(y_seg.shape))
            eval_det.update(np.sum(jec_det <= 0) / np.prod(y.shape), 1)
            csv_writter(line, 'deedsBCV')
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