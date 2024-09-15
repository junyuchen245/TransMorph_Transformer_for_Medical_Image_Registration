import os
from torch.utils.data import DataLoader
import numpy as np
import torch, glob
from TransMorph import CONFIGS as CONFIGS_TM
import TransMorph
import torch.nn.functional as F
from natsort import natsorted
import nibabel as nib
import json
from torch.utils.data import Dataset
from argparse import ArgumentParser
from scipy.ndimage import zoom
import ants
from torch import optim
import losses
import shutil
from intensity_normalization.normalize.kde import KDENormalize
from intensity_normalization.typing import Modality, TissueType

def reorient_image_to_match(reference_nii, target_nii):
    """
    Reorients the target image to match the orientation of the reference image.
    
    Args:
    target_img (str): Path to the target image that needs to be reoriented.
    reference_img (str): Path to the reference image whose orientation is the target.
    
    Returns:
    nib.Nifti1Image: The reoriented image as a Nifti object.
    """
    
    # Get the orientation of the reference image
    reference_ornt = nib.aff2axcodes(reference_nii.affine)
    
    # Reorient the target image to match the reference orientation
    target_reoriented = nib.as_closest_canonical(target_nii, enforce_diag=False)
    
    # Check current orientation of the target
    target_ornt = nib.aff2axcodes(target_reoriented.affine)
    
    # If orientations don't match, perform reorientation
    if target_ornt != reference_ornt:
        # Calculate the transformation matrix to match the reference orientation
        ornt_trans = nib.orientations.ornt_transform(nib.io_orientation(target_reoriented.affine),
                                                     nib.io_orientation(reference_nii.affine))
        
        # Apply the transformation
        target_reoriented = target_reoriented.as_reoriented(ornt_trans)
    
    return target_reoriented

class JSONDataset(Dataset):
    def __init__(self, base_dir, json_path):
        with open(json_path) as f:
            d = json.load(f)
        self.imgs = d['inputs']
        self.base_dir = base_dir

    def __getitem__(self, index):
        img_dict = self.imgs[index]
        mov_path = img_dict['moving']
        fix_path = img_dict['fixed']
        x = nib.load(self.base_dir + mov_path)
        y = nib.load(self.base_dir + fix_path)
        x = x.get_fdata() / 255.
        y = y.get_fdata() / 255.
        x, y = x[None, ...], y[None, ...]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x.float(), y.float()

    def __len__(self):
        return len(self.imgs)

def resampling(img_npy, img_pixdim, tar_pixdim, order, mode='constant'):
    if order == 0:
        img_npy = img_npy.astype(np.uint16)
    img_npy = zoom(img_npy, ((img_pixdim[0] / tar_pixdim[0]), (img_pixdim[1] / tar_pixdim[1]), (img_pixdim[2] / tar_pixdim[2])), order=order, prefilter=False, mode=mode)
    return img_npy

def intensity_norm(img_npy, mod):
    kde_norm = KDENormalize(norm_value=110)
    img_npy = kde_norm(img_npy, modality=mod)
    img_npy[img_npy < 0] = 0
    return img_npy

def save_nii(img, file_name, pix_dim=[1., 1., 1.], ref_nib=None, ref_org=None):
    if ref_nib is not None:
        x_nib = nib.Nifti1Image(img, ref_nib.affine, ref_nib.header)
        x_nib = reorient_image_to_match(ref_org, x_nib)
    else:
        x_nib = nib.Nifti1Image(img, np.eye(4))
        x_nib.header.get_xyzt_units()
        x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(file_name))

def main():
    input_dir = "./input/"
    output_dir = "./output/"
    wts_dir = "./pretrained_weights/"
    data_json = "./input_dataset.json"
    config_json = "./configs_registration.json"
    template_nib = nib.load("./template.nii.gz")
    
    
    with open(config_json) as f:
        config = json.load(f)
    if_affine = config["affine"]
    if_deformable = config["deformable"]
    if_resample = config["resample"]
    if_resample_back = config["resample_back"]
    if_instance_optimization = config["instance_optimization"]
    if_save_registration_inputs = config["save_registration_inputs"]
    if_n4_bias_correction_mov = config["n4_bias_correction_moving"]
    if_n4_bias_correction_fix = config["n4_bias_correction_fixed"]
    IO_iteration = config["IO_iteration"]
    sim_weight = None
    reg_weight = None
    verbose = config["verbose"]
    with open(data_json) as f:
        dataset_list = json.load(f)
    dataset_pairs = dataset_list['inputs']
    
    if if_deformable:
        '''
        Initialize model
        '''
        H, W, D = 160, 224, 192
        config_TM = CONFIGS_TM['TransMorph-3-LVL']
        config_TM.img_size = (H//2, W//2, D//2)
        config_TM.window_size = (H // 64, W // 64, D // 64)
        config_TM.out_chan = 3
        model = TransMorph.TransMorphTVF(config_TM, time_steps=7)
        pretrained = torch.load(wts_dir + natsorted(os.listdir(wts_dir))[0], map_location=torch.device('cpu'))
        model.load_state_dict(pretrained['state_dict'])
        print('model: {} loaded!'.format(natsorted(os.listdir(wts_dir))[0]))
        
        spatial_trans_tr = TransMorph.SpatialTransformer((H, W, D))
        spatial_trans_nn = TransMorph.SpatialTransformer((H, W, D), mode='nearest')
        
        if if_instance_optimization:
            sim_weight = config["sim_weight"]
            reg_weight = config["reg_weight"]
    
    print('"""""""""""""""""""""""""""""""""""""""""""""""""""""""\n'
          '             Registration Hyperparameters\n'
          '"""""""""""""""""""""""""""""""""""""""""""""""""""""""\n'
          'n4 bias field correction for moving image: {}\n'
          'n4 bias field correction for fixed image: {}\n'
          'affine registration: {}\n'
          'deformable (non-linear) registration: {}\n'
          'instance optimization for TransMorph: {}\n'
          'number of IO iteration: {}\n'
          'similarity measure weight: {}\n'
          'regularization weight: {}\n'
          'resampling input to match with a template: {}\n'
          'resample back to original space of the moving image: {}\n'
          '"""""""""""""""""""""""""""""""""""""""""""""""""""""""\n'.format(if_n4_bias_correction_mov, if_n4_bias_correction_fix, if_affine, if_deformable, if_instance_optimization, IO_iteration, sim_weight, reg_weight, if_resample, if_resample_back))
    
    for img_pair in dataset_pairs:
        mov_path = img_pair['moving']
        fix_path = img_pair['fixed']
        try:
            mov_mod = img_pair['moving_modality'].strip()
            fix_mod = img_pair['fixed_modality'].strip()
        except Exception:
            mov_mod = 'T1'
            fix_mod = 'T1'
            print("Imaging modalities are not provided, therefore assumed to be T1!\n")
        
        try:
            mov_bmask = img_pair['moving_brain_mask']
            fix_bmask = img_pair['fixed_brain_mask']
        except Exception:
            mov_bmask = None
            fix_bmask = None
            print("Proceed without brain masks!\n")
            
        try:
            mov_intensity_scaling_fac = img_pair['moving_scaling_factor']
            fix_intensity_scaling_fac = img_pair['fixed_scaling_factor']
        except Exception:
            mov_intensity_scaling_fac = 255
            fix_intensity_scaling_fac = 255
        
        mov_modality = Modality.T1
        if mov_mod.upper() == "T2":
            mov_modality = Modality.T2
        elif mov_mod.upper() == "PD":
            mov_modality = Modality.PD
        elif mov_mod.upper() == "FLAIR":
            mov_modality = Modality.FLAIR
            
        fix_modality = Modality.T1
        if fix_mod.upper() == "T2":
            fix_modality = Modality.T2
        elif fix_mod.upper() == "PD":
            fix_modality = Modality.PD
        elif fix_mod.upper() == "FLAIR":
            fix_modality = Modality.FLAIR
        
        mov_nib_ = nib.load(input_dir+mov_path)
        fix_nib_ = nib.load(input_dir+fix_path)
        if mov_bmask is not None:
            mov_tmp = mov_nib_.get_fdata()*nib.load(input_dir+mov_bmask).get_fdata()
            mov_nib_ = nib.Nifti1Image(mov_tmp, mov_nib_.affine, mov_nib_.header)
        if fix_bmask is not None:
            fix_tmp = fix_nib_.get_fdata()*nib.load(input_dir+fix_bmask).get_fdata()
            fix_nib_ = nib.Nifti1Image(fix_tmp, fix_nib_.affine, fix_nib_.header)
        mov_nib = reorient_image_to_match(template_nib, mov_nib_)
        fix_nib = reorient_image_to_match(template_nib, fix_nib_)

        print('moving image: {}, fixed image: {}, moving modality: {}, fixed modality: {}'.format(mov_path, fix_path, mov_mod, fix_mod))
        
        try:
            lbl_path = img_pair['label']
            lbl_nib_ = nib.load(input_dir+lbl_path)
            lbl_nib = reorient_image_to_match(template_nib, lbl_nib_)
            print('moving label: {}'.format(lbl_path))
        except Exception:
            if verbose: print('No label is given for moving image. Skipping labels...')
            lbl_nib = None
            
        '''
        Step 1: Resampling
        '''
        if verbose: print('""""""""""""""""""""""""""""""""""""""""""""\n    Step 1: Resampling    \n""""""""""""""""""""""""""""""""""""""""""""\n')
        if if_resample:
            tar_pixdim = template_nib.header.structarr['pixdim'][1:-4]
            mov_pixdim = mov_nib.header.structarr['pixdim'][1:-4]
            mov_npy = mov_nib.get_fdata()
            
            if if_n4_bias_correction_mov:
                mov_ants = ants.from_numpy(mov_npy)
                mov_ants = ants.n4_bias_field_correction(mov_ants)
                mov_npy = mov_ants.numpy()
            
            if mov_npy.max()>300:
                mov_npy = intensity_norm(mov_npy, mov_modality)
            mov_npy = resampling(mov_npy/mov_intensity_scaling_fac, mov_pixdim, tar_pixdim, order=2)
            fix_pixdim = fix_nib.header.structarr['pixdim'][1:-4]
            fix_npy = fix_nib.get_fdata()
            
            if if_n4_bias_correction_fix:
                fix_ants = ants.from_numpy(fix_npy)
                fix_ants = ants.n4_bias_field_correction(fix_ants)
                fix_npy = fix_ants.numpy()
            
            if fix_npy.max()>300:
                fix_npy = intensity_norm(fix_npy, fix_modality)
            fix_npy = resampling(fix_npy/fix_intensity_scaling_fac, fix_pixdim, tar_pixdim, order=2)
            
            if lbl_nib is not None:
                lbl_npy = lbl_nib.get_fdata()
                lbl_npy = resampling(lbl_npy, mov_pixdim, tar_pixdim, order=0, mode='nearest')
            if verbose: print('----Step 1: Done! \n')
        else:
            if verbose: print('----Skipping Step 1!\n')
            mov_npy = mov_nib.get_fdata()
            fix_npy = fix_nib.get_fdata()
            
            if if_n4_bias_correction_mov:
                mov_ants = ants.from_numpy(mov_npy)
                mov_ants = ants.n4_bias_field_correction(mov_ants)
                mov_npy = mov_ants.numpy()
            
            if if_n4_bias_correction_fix:
                fix_ants = ants.from_numpy(fix_npy)
                fix_ants = ants.n4_bias_field_correction(fix_ants)
                fix_npy = fix_ants.numpy()
            
            if mov_npy.max()>300:
                mov_npy = intensity_norm(mov_npy, mov_modality)
            mov_npy = mov_npy/mov_intensity_scaling_fac
            if fix_npy.max()>300:
                fix_npy = intensity_norm(fix_npy, fix_modality)
            fix_npy = fix_npy/fix_intensity_scaling_fac
            if lbl_nib is not None:
                lbl_npy = lbl_nib.get_fdata()
        if verbose: print('----Moving image {}, Fixed image {}'.format(mov_npy.shape, fix_npy.shape))
        
        '''
        Step 2: Affine Registration
        '''
        if verbose: print('""""""""""""""""""""""""""""""""""""""""""""\n    Step 2: Affine Registration    \n""""""""""""""""""""""""""""""""""""""""""""\n')
        if if_affine:
            tmp_npy = template_nib.get_fdata()/255
            tmp_ants = ants.from_numpy(tmp_npy)
            mov_ants = ants.from_numpy(mov_npy)
            fix_ants = ants.from_numpy(fix_npy)
            regMovTmp = ants.registration(fixed=tmp_ants, moving=mov_ants, type_of_transform='Affine', aff_metric='mattes')
            mov_ants = ants.apply_transforms(fixed=tmp_ants, moving=mov_ants, transformlist=regMovTmp['fwdtransforms'],)
            regFixTmp = ants.registration(fixed=tmp_ants, moving=fix_ants, type_of_transform='Affine', aff_metric='mattes')
            fix_ants = ants.apply_transforms(fixed=tmp_ants, moving=fix_ants, transformlist=regFixTmp['fwdtransforms'],)
            
            if lbl_nib is not None:
                lbl_ants = ants.from_numpy(lbl_npy)
                lbl_ants = ants.apply_transforms(fixed=tmp_ants, moving=lbl_ants, transformlist=regMovTmp['fwdtransforms'], interpolator="nearestNeighbor")
                lbl_npy = lbl_ants.numpy()
                
            mov_npy = mov_ants.numpy()
            fix_npy = fix_ants.numpy()
            if verbose: print('----Step 2: Done!\n')
            if verbose: print('----Affine aligned moving image {}, Fixed image {}'.format(mov_npy.shape, fix_npy.shape))
        else:
            def_mov_npy = mov_ants.numpy()
            if verbose: print('----Skipping Step 2!\n')
        
        '''
        Step 3: Deformable Registration
        '''
        if verbose: print('""""""""""""""""""""""""""""""""""""""""""""\n    Step 3: Deformable Registration    \n""""""""""""""""""""""""""""""""""""""""""""\n')
        if if_deformable:
            mov_torch = torch.from_numpy(mov_npy[None, None, ...])
            fix_torch = torch.from_numpy(fix_npy[None, None, ...])
            x_half = F.avg_pool3d(mov_torch, 2)
            y_half = F.avg_pool3d(fix_torch, 2)
            if if_instance_optimization:
                criterion_sim = losses.NCC_vxm()
                criterion_reg = losses.Grad3d(penalty='l2')
                model.load_state_dict(pretrained['state_dict'])
                if verbose: print('IO initiated!\n Starting from the pretrained model: {}\n'.format(natsorted(os.listdir(wts_dir))[0]))
                
                model.train()
                optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0, amsgrad=True)
                for iter_ in range(IO_iteration):
                    flow = model((x_half, y_half))
                    flow = F.interpolate(flow, scale_factor=2, mode='trilinear', align_corners=False) * 2
                    output = spatial_trans_tr(mov_torch, flow)
                    loss_ncc = criterion_sim(output, fix_torch) * sim_weight
                    loss_reg = criterion_reg(flow, fix_torch) * reg_weight
                    loss = loss_ncc + loss_reg
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if verbose: print("     Instance Optimization: Iteration-{} of {}, Loss-{:.6f}, Sim-{:.6f}, Reg-{:.6f}".format(iter_+1, IO_iteration, loss, loss_ncc, loss_reg))
                    
            with torch.no_grad():
                model.eval()
                flow = model((x_half, y_half))
                flow = F.interpolate(flow, scale_factor=2, mode='trilinear', align_corners=False) * 2
             
                def_mov_torch = spatial_trans_tr(mov_torch, flow)
                def_mov_npy = def_mov_torch.detach().cpu().numpy()[0, 0]
                if lbl_nib is not None:
                    lbl_torch = torch.from_numpy(lbl_npy[None, None, ...])
                    def_lbl_torch = spatial_trans_nn(lbl_torch, flow)
                    def_lbl_npy = def_lbl_torch.detach().cpu().numpy()[0, 0]
            if verbose: print('----Step 3: Done!\n')
        else:
            if verbose: print('----Skipping Step 3!\n')
        
        '''
        Step 4: Resampling Back
        ''' 
        if verbose: print('""""""""""""""""""""""""""""""""""""""""""""\n    Step 4: Resampling Back to Original    \n""""""""""""""""""""""""""""""""""""""""""""\n')      
        if if_resample_back:
            if if_affine:
                def_mov_ants = ants.from_numpy(def_mov_npy)
                def_mov_ants_2mov = ants.apply_transforms(fixed=tmp_ants, moving=def_mov_ants, transformlist=regMovTmp['fwdtransforms'], whichtoinvert=[True,],)
                def_mov_movorg_npy = def_mov_ants_2mov.numpy()
                def_mov_ants_2fix = ants.apply_transforms(fixed=tmp_ants, moving=def_mov_ants, transformlist=regFixTmp['fwdtransforms'], whichtoinvert=[True,],)
                def_mov_fixorg_npy = def_mov_ants_2fix.numpy()
                if lbl_nib is not None:
                    def_lbl_ants = ants.from_numpy(def_lbl_npy)
                    def_lbl_ants_2mov = ants.apply_transforms(fixed=tmp_ants, moving=def_lbl_ants, transformlist=regMovTmp['fwdtransforms'], whichtoinvert=[True,], interpolator="nearestNeighbor")
                    def_lbl_movorg_npy = def_lbl_ants_2mov.numpy()
                    def_lbl_ants_2fix = ants.apply_transforms(fixed=tmp_ants, moving=def_lbl_ants, transformlist=regFixTmp['fwdtransforms'], whichtoinvert=[True,], interpolator="nearestNeighbor")
                    def_lbl_fixorg_npy = def_lbl_ants_2fix.numpy()
            
            if if_resample:
                def_mov_movorg_npy = resampling(def_mov_movorg_npy, tar_pixdim, mov_pixdim, order=2)*mov_intensity_scaling_fac
                def_mov_fixorg_npy = resampling(def_mov_fixorg_npy, tar_pixdim, fix_pixdim, order=2)*mov_intensity_scaling_fac
                if lbl_nib is not None:
                    def_lbl_movorg_npy = resampling(def_lbl_movorg_npy, tar_pixdim, mov_pixdim, order=0)
                    def_lbl_fixorg_npy = resampling(def_lbl_fixorg_npy, tar_pixdim, fix_pixdim, order=0)
            if verbose: print('----Step 4: Done!\n')
        else:
            if verbose: print('----Skipping Step 4!\n')
        
        '''
        Step 5: Saving files
        '''
        if verbose: print('""""""""""""""""""""""""""""""""""""""""""""\n    Step 5: Saving Files    \n""""""""""""""""""""""""""""""""""""""""""""\n')  
        mov_name = mov_path.split('/')[-1].split('.nii')[0]
        fix_name = fix_path.split('/')[-1].split('.nii')[0]
        folder_name = 'mov_{}_fix_{}/'.format(mov_name, fix_name)
        if not os.path.exists(output_dir+folder_name):
            os.makedirs(output_dir+folder_name)
        
        save_nii(def_mov_npy*mov_intensity_scaling_fac, output_dir+folder_name+'deformed_moving_image', tar_pixdim)
        if if_resample_back:
            save_nii(def_mov_movorg_npy, output_dir+folder_name+'deformed_moving_image_original_moving_space', mov_pixdim, mov_nib, mov_nib_)
            save_nii(def_mov_fixorg_npy, output_dir+folder_name+'deformed_moving_image_original_fixed_space', fix_pixdim, fix_nib, fix_nib_)
            
        if if_save_registration_inputs:
            save_nii(fix_npy*fix_intensity_scaling_fac, output_dir+folder_name+'fixed_image_final', tar_pixdim)
            save_nii(mov_npy*mov_intensity_scaling_fac, output_dir+folder_name+'moving_image_final', tar_pixdim)
            mov_nib.to_filename(output_dir+folder_name+'moving_image_reoriented.nii.gz')
            fix_nib.to_filename(output_dir+folder_name+'fixed_image_reoriented.nii.gz')
            if lbl_nib is not None:
                save_nii(lbl_npy, output_dir+folder_name+'moving_label_final', tar_pixdim)
                lbl_nib.to_filename(output_dir+folder_name+'moving_label_reoriented.nii.gz')
            
        if if_deformable:
            flow = flow.cpu().detach().numpy()[0]
            save_nii(flow, output_dir+folder_name+'displacement_field', tar_pixdim)
        if if_affine:
            shutil.copyfile(regMovTmp['fwdtransforms'][0], output_dir+folder_name+ "affine_fwdtransforms.mat")
        if lbl_nib is not None:
            save_nii(def_lbl_npy, output_dir+folder_name+'deformed_moving_label', tar_pixdim)
            if if_resample_back:
                save_nii(def_lbl_movorg_npy, output_dir+folder_name+'deformed_moving_label_original_moving_space', mov_pixdim, mov_nib, mov_nib_)
                save_nii(def_lbl_fixorg_npy, output_dir+folder_name+'deformed_moving_label_original_fixed_space', fix_pixdim, fix_nib, fix_nib_)
        if verbose: print('----Step 5: Done!\n')
        
if __name__ == '__main__':
    torch.manual_seed(0)
    main()
