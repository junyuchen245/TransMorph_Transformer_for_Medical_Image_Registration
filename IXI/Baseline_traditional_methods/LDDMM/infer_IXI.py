import glob
import os, utils, torch
import sys
from torch.utils.data import DataLoader
from data_IXI import datasets, trans
import matplotlib.pyplot as plt
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
    csv_writter(line + ',' + 'non_jec', 'lddmm_IXI')
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
            dx = [1, 1, 1]
            lddmm = torch_lddmm.LDDMM(template=x * 255., target=y * 255., outdir='./', do_affine=0, do_lddmm=1, a=5.,
                                      p=2,
                                      niter=500, epsilon=2, sigma=4., sigmaR=3.2, optimizer='sgd', dx=dx, nt=7,
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
            csv_writter(line, 'lddmm_IXI')
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(y_seg.shape), 1)
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