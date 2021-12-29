import glob
import os, utils, torch
from torch.utils.data import DataLoader
from data_IXI import datasets, trans
import matplotlib.pyplot as plt
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

def detJacobian(Axij,Ayij, img_sz, **kwargs):
    [xFX, xFY] = np.gradient(Axij)
    [yFX, yFY] = np.gradient(Ayij)
    jac_det = np.zeros(Axij.shape)
    for i in range(img_sz):
        for j in range(img_sz):
            jac_mij = [[xFX[i, j], xFY[i, j]], [yFX[i, j], yFY[i, j]]]
            jac_det[i, j] =  np.linalg.det(jac_mij)
    return jac_det

def plot_grid(gridx,gridy, **kwargs):
    for i in range(gridx.shape[1]):
        plt.plot(gridx[i,:], gridy[i,:], linewidth=0.8, **kwargs)
    for i in range(gridx.shape[0]):
        plt.plot(gridx[:,i], gridy[:,i], linewidth=0.8, **kwargs)

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

def MSE_torch(x, y):
    return np.mean((x - y) ** 2)

def MAE_torch(x, y):
    return np.mean(np.abs(x - y))

def main():
    atlas_dir = 'Path_to_IXI_data/atlas.pkl'
    test_dir = 'Path_to_IXI_data/Val/'
    dict = utils.process_label()
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line + ',' + 'non_jec', 'NiftyReg_IXI')
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    stdy_idx = 0
    eval_dsc_def = AverageMeter()
    eval_det = AverageMeter()
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

            os.system('/mnt/d/pythonProject/NiftiReg/nifty_reg/niftyreg_install/bin/reg_f3d -be 0.0006 --ssd -ref y.nii.gz -flo x.nii.gz -res output_deformed.nii.gz -cpp ref_template_flo_new_image_nrr_cpp.nii')
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
            csv_writter(line, 'NiftyReg_IXI')
            eval_dsc_def.update(dsc_trans.item(), 1)
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def visual_flow(flow, down = 7):
    flow = flow.detach().cpu().numpy()[0, :, :, :, :]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = np.meshgrid(np.arange(0, flow.shape[2], 1),
                          np.arange(0, flow.shape[1], 1),
                          np.arange(0, flow.shape[3], 1))
    print(x.shape)

    xdown = x[0:-1:down, 0:-1:down, 0:-1:down]
    ydown = y[0:-1:down, 0:-1:down, 0:-1:down]
    zdown = z[0:-1:down, 0:-1:down, 0:-1:down]
    u = flow[0, :, :, :]
    udown = u[0:-1:down, 0:-1:down, 0:-1:down]
    print(u.shape)
    v = flow[1, :, :, :]
    vdown = v[0:-1:down, 0:-1:down, 0:-1:down]
    w = flow[2, :, :, :]
    wdown = w[0:-1:down, 0:-1:down, 0:-1:down]

    ax.quiver(xdown, ydown, zdown, udown, vdown, wdown, length=2, linewidths=1)
    plt.show()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 16:32, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def display_deformed_grid(flow, img_sz=160, down=7):
    flow = flow.detach().cpu().numpy()[0, :, 76:80, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=200)
    for idx in range(flow.shape[1]):
        x = np.arange(0, img_sz, 1)
        y = np.arange(0, img_sz, 1)
        X, Y = np.meshgrid(x, y)
        u =flow[1, idx, :, :].reshape(img_sz, img_sz)
        v =flow[2, idx, :, :].reshape(img_sz, img_sz)
        #print('max u: '+str(np.max(u)))

        phix = X; phiy = Y
        for i in range(0, img_sz):
            for j in range(0, img_sz):
                # add the displacement for each p(k) in the sum
                phix[i, j] = phix[i, j] - u[i, j]
                phiy[i, j] = phiy[i, j] - v[i, j]
        phixdown = phix[0:-1:down, 0:-1:down]
        phiydown = phiy[0:-1:down, 0:-1:down]
        ax = plt.subplot(2, 2, idx + 1)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')
        plot_grid(phixdown,phiydown, color="k")
        plt.gca().invert_yaxis()
        #plt.subplot(1, 7, 2)
        #detJac = detJacobian(phix, phiy, img_sz)
        #print('Min det(Jac): '+str(np.min(np.abs(detJac))))
        #print('# det(Jac)<=0: ' + str((detJac == 0).sum()))
        #plt.imshow(detJac); plt.title('det(Jacobian)')
        #plt.colorbar()
    plt.show()
    return fig

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    main()