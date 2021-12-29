import glob
import os, utils, torch
import sys, ants
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
    csv_writter(line+','+'non_jec', 'ants_IXI')
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    stdy_idx = 0
    eval_dsc_def = AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
            y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
            x_seg = data[2].squeeze(0).squeeze(0).detach().cpu().numpy()
            y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()
            x = ants.from_numpy(x)
            y = ants.from_numpy(y)

            x_ants = ants.from_numpy(x_seg.astype(np.float32))
            y_ants = ants.from_numpy(y_seg.astype(np.float32))

            reg12 = ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')
            def_seg = ants.apply_transforms(fixed=y_ants,
                                            moving=x_ants,
                                            transformlist=reg12['fwdtransforms'],
                                            interpolator='nearestNeighbor',)
                                            #whichtoinvert=[True, False, True, False]

            flow = np.array(nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
            flow = flow[:,:,:,0,:].transpose(3, 0, 1, 2)
            def_seg = def_seg.numpy()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            y_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_trans = utils.dice_val(def_seg.long(), y_seg.long(), 46)
            eval_dsc_def.update(dsc_trans.item(), 1)
            jac_det = utils.jacobian_determinant_vxm(flow)
            line = utils.dice_val_substruct(def_seg.long(), y_seg.long(), stdy_idx)
            line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(y_seg.shape))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            csv_writter(line, 'ants_IXI')
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            stdy_idx += 1
        print('Deformed DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))

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