import torch
import numpy as np
import scipy.linalg
import time
import sys
import os
import distutils.version
sys.path.insert(0,'/cis/home/leebc/Software/')
import nibabel as nib

def mygaussian(sigma=1,size=5):
    ind = np.linspace(-np.floor(size/2.0),np.floor(size/2.0),size)
    X,Y = np.meshgrid(ind,ind,indexing='xy')
    out_mat = np.exp(-(X**2 + Y**2) / (2*sigma**2))
    out_mat = out_mat / np.sum(out_mat)
    return out_mat

def mygaussian_torch_selectcenter(sigma=1,center_x=0,center_y=0,size_x=100,size_y=100):
    ind_x = torch.linspace(0,size_x-1,size_x)
    ind_y = torch.linspace(0,size_y-1,size_y)
    X,Y = torch.meshgrid(ind_x,ind_y)
    out_mat = torch.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2*sigma**2))
    #out_mat = out_mat / torch.sum(out_mat)
    return out_mat

def mygaussian_torch_selectcenter_meshgrid(X,Y,sigma=1,center_x=0,center_y=0):
    out_mat = torch.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2*sigma**2))
    #out_mat = out_mat / torch.sum(out_mat)
    return out_mat

def mygaussian3d(sigma=1,size=5):
    ind = np.linspace(-np.floor(size/2.0),np.floor(size/2.0),size)
    X,Y,Z = np.meshgrid(ind,ind,ind,indexing='xy')
    out_mat = np.exp(-(X**2 + Y**2 + Z**2) / (2*sigma**2))
    out_mat = out_mat / np.sum(out_mat)
    return out_mat

def mygaussian_3d_torch_selectcenter_meshgrid(X,Y,Z,sigma=1,center_x=0,center_y=0,center_z=0):
    out_mat = torch.exp(-((X-center_x)**2 + (Y-center_y)**2 + (Z-center_z)**2) / (2*sigma**2))
    #out_mat = out_mat / torch.sum(out_mat)
    return out_mat

def grid_sample(*args,**kwargs):
    if distutils.version.LooseVersion(torch.__version__) < distutils.version.LooseVersion("1.3.0"):
        return torch.nn.functional.grid_sample(*args,**kwargs)
    else:
        return torch.nn.functional.grid_sample(*args,**kwargs,align_corners=True)

class LDDMM:
    def __init__(self,template=None,target=None,costmask=None,outdir='./',gpu_number=0,a=5.0,p=2,niter=100,epsilon=5e-3,epsilonL=1.0e-7,epsilonT=2.0e-5,sigma=2.0,sigmaR=1.0,nt=5,do_lddmm=1,do_affine=0,checkaffinestep=0,optimizer='gd',sg_mask_mode='ones',sg_rand_scale=1.0,sg_sigma=1.0,sg_climbcount=1,sg_holdcount=1,sg_gamma=0.9,adam_alpha=0.1,adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-8,ada_rho=0.95,ada_epsilon=1e-6,rms_rho=0.9,rms_epsilon=1e-8,rms_alpha=0.001,maxclimbcount=3,savebestv=False,minenergychange = 0.000001,minbeta=1e-4,dtype='float',im_norm_ms=0,slice_alignment=0,energy_fraction=0.02,energy_fraction_from=0,cc=0,cc_channels=[],we=0,we_channels=[],sigmaW=1.0,nMstep=5,dx=None,low_memory=0,update_epsilon=0,verbose=1,v_scale=1.0,v_scale_smoothing=0):
        self.params = {}
        self.params['gpu_number'] = gpu_number
        self.params['a'] = float(a)
        self.params['p'] = float(p)
        self.params['niter'] = niter
        self.params['epsilon'] = float(epsilon)
        self.params['epsilonL'] = float(epsilonL)
        self.params['epsilonT'] = float(epsilonT)
        if isinstance(sigma,(int,float)):
            self.params['sigma'] = float(sigma)
        else:
            self.params['sigma'] = [float(x) for x in sigma]
        self.params['sigmaR'] = float(sigmaR)
        self.params['nt'] = nt
        self.params['orig_nt'] = nt
        self.params['template'] = template
        self.params['target'] = target
        self.params['costmask'] = costmask
        self.params['outdir'] = outdir
        self.params['do_lddmm'] = do_lddmm
        self.params['do_affine'] = do_affine
        self.params['checkaffinestep'] = checkaffinestep
        self.params['optimizer'] = optimizer
        self.params['sg_sigma'] = float(sg_sigma)
        self.params['sg_mask_mode'] = sg_mask_mode
        self.params['sg_rand_scale'] = float(sg_rand_scale)
        self.params['sg_climbcount'] = int(sg_climbcount)
        self.params['sg_holdcount'] = int(sg_holdcount)
        self.params['sg_gamma'] = float(sg_gamma)
        self.params['adam_alpha'] = float(adam_alpha)
        self.params['adam_beta1'] = float(adam_beta1)
        self.params['adam_beta2'] = float(adam_beta2)
        self.params['adam_epsilon'] = float(adam_epsilon)
        self.params['ada_rho'] = float(ada_rho)
        self.params['ada_epsilon'] = float(ada_epsilon)
        self.params['rms_rho'] = float(rms_rho)
        self.params['rms_epsilon'] = float(rms_epsilon)
        self.params['rms_alpha'] = float(rms_alpha)
        self.params['maxclimbcount'] = maxclimbcount
        self.params['savebestv'] = savebestv
        self.params['minbeta'] = minbeta
        self.params['minenergychange'] = minenergychange
        self.params['im_norm_ms'] = im_norm_ms
        self.params['slice_alignment'] = slice_alignment
        self.params['energy_fraction'] = energy_fraction
        self.params['energy_fraction_from'] = energy_fraction_from
        self.params['cc'] = cc
        self.params['cc_channels'] = cc_channels
        self.params['we'] = we
        self.params['we_channels'] = we_channels
        self.params['sigmaW'] = sigmaW
        self.params['nMstep'] = nMstep
        self.params['v_scale'] = float(v_scale)
        self.params['v_scale_smoothing'] = int(v_scale_smoothing)
        self.params['dx'] = dx
        dtype_dict = {}
        dtype_dict['float'] = 'torch.FloatTensor'
        dtype_dict['double'] = 'torch.DoubleTensor'
        self.params['dtype'] = dtype_dict[dtype]
        self.params['low_memory'] = low_memory
        self.params['update_epsilon'] = float(update_epsilon)
        self.params['verbose'] = float(verbose)
        optimizer_dict = {}
        optimizer_dict['gd'] = 'gradient descent'
        optimizer_dict['gdr'] = 'gradient descent with reducing epsilon'
        optimizer_dict['gdw'] = 'gradient descent with delayed reducing epsilon'
        optimizer_dict['adam'] = 'adaptive moment estimation (UNDER CONSTRUCTION)'
        optimizer_dict['adadelta'] = 'adadelta (UNDER CONSTRUCTION)'
        optimizer_dict['rmsprop'] = 'root mean square propagation (UNDER CONSTRUCTION)'
        optimizer_dict['sgd'] = 'stochastic gradient descent'
        optimizer_dict['sgdm'] = 'stochastic gradient descent with momentum (UNDER CONSTRUCTION)'
        print('\nCurrent parameters:')
        print('>    a               = ' + str(a) + ' (smoothing kernel, a*(pixel_size))')
        print('>    p               = ' + str(p) + ' (smoothing kernel power, p*2)')
        print('>    niter           = ' + str(niter) + ' (number of iterations)')
        print('>    epsilon         = ' + str(epsilon) + ' (gradient descent step size)')
        print('>    epsilonL        = ' + str(epsilonL) + ' (gradient descent step size, affine)')
        print('>    epsilonT        = ' + str(epsilonT) + ' (gradient descent step size, translation)')
        print('>    minbeta         = ' + str(minbeta) + ' (smallest multiple of epsilon)')
        print('>    sigma           = ' + str(sigma) + ' (matching term coefficient (0.5/sigma**2))')
        print('>    sigmaR          = ' + str(sigmaR)+ ' (regularization term coefficient (0.5/sigmaR**2))')
        print('>    nt              = ' + str(nt) + ' (number of time steps in velocity field)')
        print('>    do_lddmm        = ' + str(do_lddmm) + ' (perform LDDMM step, 0 = no, 1 = yes)')
        print('>    do_affine       = ' + str(do_affine) + ' (interleave linear registration: 0 = no, 1 = affine, 2 = rigid)')
        print('>    checkaffinestep = ' + str(checkaffinestep) + ' (evaluate linear matching energy: 0 = no, 1 = yes)')
        print('>    im_norm_ms      = ' + str(im_norm_ms) + ' (normalize image by mean and std: 0 = no, 1 = yes)')
        print('>    gpu_number      = ' + str(gpu_number) + ' (index of CUDA_VISIBLE_DEVICES to use)')
        print('>    dtype           = ' + str(dtype) + ' (bit depth, \'float\' or \'double\')')
        print('>    energy_fraction = ' + str(energy_fraction) + ' (fraction of initial energy at which to stop)')
        print('>    cc              = ' + str(cc) + ' (contrast correction: 0 = no, 1 = yes)')
        print('>    cc_channels     = ' + str(cc_channels) + ' (image channels to run contrast correction (0-indexed))')
        print('>    we              = ' + str(we) + ' (weight estimation: 0 = no, 2+ = yes)')
        print('>    we_channels     = ' + str(we_channels) + ' (image channels to run weight estimation (0-indexed))')
        print('>    sigmaW          = ' + str(sigmaW) + ' (coefficient for each weight estimation class)')
        print('>    nMstep          = ' + str(nMstep) + ' (update weight estimation every nMstep steps)')
        print('>    v_scale         = ' + str(v_scale) + ' (parameter scaling factor)')
        if v_scale < 1.0:
            print('>    v_scale_smooth  = ' + str(v_scale_smoothing) + ' (smoothing before interpolation for v-scaling: 0 = no, 1 = yes)')
        print('>    low_memory      = ' + str(low_memory) + ' (low memory mode: 0 = no, 1 = yes)')
        print('>    update_epsilon  = ' + str(update_epsilon) + ' (update optimization step size between runs: 0 = no, 1 = yes)')
        print('>    outdir          = ' + str(outdir) + ' (output directory name)')
        if optimizer in optimizer_dict:
            print('>    optimizer       = ' + str(optimizer_dict[optimizer]) + ' (optimizer type)')
            if optimizer == 'adam':
                print('>    +adam_alpha     = ' + str(adam_alpha) + ' (learning rate)')
                print('>    +adam_beta1     = ' + str(adam_beta1) + ' (decay rate 1)')
                print('>    +adam_beta2     = ' + str(adam_beta2) + ' (decay rate 2)')
                print('>    +adam_epsilon   = ' + str(adam_epsilon) + ' (epsilon)')
                print('>    +sg_sigma       = ' + str(sg_sigma) + ' (subsampler sigma (for gaussian mode))')
                print('>    +sg_mask_mode   = ' + str(sg_mask_mode) + ' (subsampler scheme)')
                print('>    +sg_climbcount  = ' + str(sg_climbcount) + ' (# of times energy is allowed to increase)')
                print('>    +sg_holdcount   = ' + str(sg_holdcount) + ' (# of iterations per random mask)')
                print('>    +sg_rand_scale  = ' + str(sg_rand_scale) + ' (scale for non-gauss sg masking)')
            elif optimizer == "adadelta":
                print('>    +ada_rho        = ' + str(ada_rho) + ' (decay rate)')
                print('>    +ada_epsilon    = ' + str(ada_epsilon) + ' (epsilon)')
            elif optimizer == "rmsprop":
                print('>    +rms_rho        = ' + str(rms_rho) + ' (decay rate)')
                print('>    +rms_epsilon    = ' + str(rms_epsilon) + ' (epsilon)')
                print('>    +rms_alpha      = ' + str(rms_alpha) + ' (learning rate)')
                print('>    +sg_sigma       = ' + str(sg_sigma) + ' (subsampler sigma (for gaussian mode))')
                print('>    +sg_mask_mode   = ' + str(sg_mask_mode) + ' (subsampler scheme)')
                print('>    +sg_climbcount  = ' + str(sg_climbcount) + ' (# of times energy is allowed to increase)')
                print('>    +sg_holdcount   = ' + str(sg_holdcount) + ' (# of iterations per random mask)')
                print('>    +sg_rand_scale  = ' + str(sg_rand_scale) + ' (scale for non-gauss sg masking)')
            elif optimizer == 'sgd' or optimizer == 'sgdm':
                print('>    +sg_sigma       = ' + str(sg_sigma) + ' (subsampler sigma (for gaussian mode))')
                print('>    +sg_mask_mode   = ' + str(sg_mask_mode) + ' (subsampler scheme)')
                print('>    +sg_climbcount  = ' + str(sg_climbcount) + ' (# of times energy is allowed to increase)')
                print('>    +sg_holdcount   = ' + str(sg_holdcount) + ' (# of iterations per random mask)')
                print('>    +sg_rand_scale  = ' + str(sg_rand_scale) + ' (scale for non-gauss sg masking)')
                if optimizer == 'sgdm':
                    print('>    +sg_gamma       = ' + str(sg_gamma) + ' (fraction of paste updates)')
        
        else:
            print('WARNING: optimizer \'' + str(optimizer) + '\' not recognized. Setting to basic gradient descent with reducing step size.')
            self.params['optimizer'] = 'gdr'
        
        print('\n')
        if template is None:
            print('WARNING: template file name is not set. Use LDDMM.setParams(\'template\',filename\/array).\n')
        elif isinstance(template,np.ndarray):
            print('>    template        = numpy.ndarray\n')
        elif isinstance(template,list) and isinstance(template[0],np.ndarray):
            myprintstring = '>    template        = [numpy.ndarray'
            for i in range(len(template)-1):
                myprintstring = myprintstring + ', numpy.ndarray'
            
            myprintstring = myprintstring + ']\n'
            print(myprintstring)
        else:
            print('>    template        = ' + str(template) + '\n')
        
        if target is None:
            print('WARNING: target file name is not set. Use LDDMM.setParams(\'target\',filename\/array).\n')
        elif isinstance(target,np.ndarray):
            print('>    target          = numpy.ndarray\n')
        elif isinstance(target,list) and isinstance(target[0],np.ndarray):
            myprintstring = '>    target          = [numpy.ndarray'
            for i in range(len(target)-1):
                myprintstring = myprintstring + ', numpy.ndarray'
            
            myprintstring = myprintstring + ']\n'
            print(myprintstring)
        else:
            print('>    target          = ' + str(target) + '\n')
        
        if isinstance(costmask,np.ndarray):
            print('>    costmask        = numpy.ndarray (costmask file name or numpy.ndarray)')
        else:
            print('>    costmask        = ' + str(costmask) + ' (costmask file name or numpy.ndarray)')
        
        self.initializer_flags = {}
        self.initializer_flags['load'] = 1
        self.initializer_flags['v_scale'] = 0
        if self.params['do_lddmm'] == 1:
            self.initializer_flags['lddmm'] = 1
        else:
            self.initializer_flags['lddmm'] = 0
        
        if self.params['do_affine'] > 0:
            self.initializer_flags['affine'] = 1
        else:
            self.initializer_flags['affine'] = 0
        
        if self.params['cc'] != 0:
            self.initializer_flags['cc'] = 1
        else:
            self.initializer_flags['cc'] = 0
        
        if self.params['we'] >= 2:
            self.initializer_flags['we'] = 1
        else:
            self.initializer_flags['we'] = 0
    
    # manual edit parameter
    def setParams(self,parameter_name,parameter_value):
        if parameter_name in self.params:
            print('Parameter \'' + str(parameter_name) + '\' changed to \'' + str(parameter_value) + '\'.')
            if parameter_name == 'template' or parameter_name == 'target' or parameter_name == 'costmask':
                self.initializer_flags['load'] = 1
            elif parameter_name == 'do_lddmm' and parameter_value == 1 and self.params['do_lddmm'] == 0:
                self.initializer_flags['lddmm'] = 1
                print('WARNING: LDDMM state has changed. Variables will be initialized.')
            elif parameter_name == 'do_affine' and parameter_value > 0 and self.params['do_affine'] != parameter_value:
                self.initializer_flags['affine'] = 1
                print('WARNING: Affine state has changed. Variables will be initialized.')
            elif (parameter_name == 'cc' and parameter_value != 0 and self.params['cc'] != parameter_value) or (parameter_name == 'cc_channels' and parameter_value != self.params['cc_channels']):
                self.initializer_flags['cc'] = 1
                print('WARNING: Contrast correction state has changed. Variables will be initialized.')
            elif parameter_name == 'we' and parameter_value >= 2 and self.params['we'] != parameter_value or (parameter_name == 'we_channels' and parameter_value != self.params['we_channels']):
                self.initializer_flags['we'] = 1
                print('WARNING: Weight estimation state has changed. Variables will be initialized.')
            elif parameter_name == 'v_scale' and self.params['do_lddmm'] == 1 and hasattr(self,'vt0'):
                self.initializer_flags['v_scale'] = 1
                print('WARNING: Parameter sparsity has changed. Variables will be initialized.')
            
            self.params[parameter_name] = parameter_value
        else:
            print('Parameter \'' + str(parameter_name) + '\' is not a valid parameter.')
        
        return
    
    # image loader
    def loadImage(self, filename,im_norm_ms=0):
        fname, fext = os.path.splitext(filename)
        if fext == '.img' or fext == '.hdr':
            img_struct = nib.load(fname + '.img')
            spacing = img_struct.header['pixdim'][1:4]
            size = img_struct.header['dim'][1:4]
            image = np.squeeze(img_struct.get_data().astype(np.float32))
            if im_norm_ms == 1:
                if np.std(image) != 0:
                    image = torch.tensor((image - np.mean(image)) / np.std(image)).type(self.params['dtype']).to(device=self.params['cuda'])

                else:
                    image = torch.tensor((image - np.mean(image)) ).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('WARNING: stdev of image is zero, not rescaling.')
            else:
                image = torch.tensor(image).type(self.params['dtype']).to(device=self.params['cuda'])
            return (image, spacing, size)
        elif fext == '.nii':
            img_struct = nib.load(fname + '.nii')
            spacing = img_struct.header['pixdim'][1:4]
            size = img_struct.header['dim'][1:4]
            image = np.squeeze(img_struct.get_data().astype(np.float32))
            if im_norm_ms == 1:
                if np.std(image) != 0:
                    image = torch.tensor((image - np.mean(image)) / np.std(image)).type(self.params['dtype']).to(device=self.params['cuda'])

                else:
                    image = torch.tensor((image - np.mean(image)) ).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('WARNING: stdev of image is zero, not rescaling.')
            else:
                image = torch.tensor(image).type(self.params['dtype']).to(device=self.params['cuda'])
            return (image, spacing, size)
        else:
            print('File format not supported.\n')
            return (-1,-1,-1)
    
    # helper function to check parameters before running registration
    def _checkParameters(self):
        flag = 1
        if self.params['gpu_number'] is not None and not isinstance(self.params['gpu_number'], (int, float)):
            flag = -1
            print('ERROR: gpu_number must be None or a number.')
        else:
            if self.params['gpu_number'] is None:
                self.params['cuda'] = 'cpu'
            else:
                self.params['cuda'] = 'cuda:' + str(self.params['gpu_number'])
        
        number_list = ['a','p','niter','epsilon','sigmaR','nt','do_lddmm','do_affine','epsilonL','epsilonT','im_norm_ms','slice_alignment','energy_fraction','energy_fraction_from','cc','we','nMstep','low_memory','update_epsilon','v_scale','adam_alpha','adam_beta1','adam_beta2','adam_epsilon','ada_rho','ada_epsilon','rms_rho','rms_alpha','rms_epsilon','sg_sigma','sg_climbcount','sg_rand_scale','sg_holdcount','sg_gamma','v_scale_smoothing','verbose']
        string_list = ['outdir','optimizer']
        stringornone_list = ['costmask'] # or array, actually
        stringorlist_list = ['template','target'] # or array, actually
        numberorlist_list = ['sigma','cc_channels','we_channels','sigmaW']
        noneorarrayorlist_list = ['dx']
        for i in range(len(number_list)):
            if not isinstance(self.params[number_list[i]], (int, float)):
                flag = -1
                print('ERROR: ' + number_list[i] + ' must be a number.')
        
        for i in range(len(string_list)):
            if not isinstance(self.params[string_list[i]], str):
                flag = -1
                print('ERROR: ' + string_list[i] + ' must be a string.')
        
        for i in range(len(stringornone_list)):
            if not isinstance(self.params[stringornone_list[i]], (str,np.ndarray)) and self.params[stringornone_list[i]] is not None:
                flag = -1
                print('ERROR: ' + stringornone_list[i] + ' must be a string or None.')
        
        for i in range(len(stringorlist_list)):
            if not isinstance(self.params[stringorlist_list[i]], str) and not isinstance(self.params[stringorlist_list[i]], list) and not isinstance(self.params[stringorlist_list[i]], np.ndarray):
                flag = -1
                print('ERROR: ' + stringorlist_list[i] + ' must be a string or an np.ndarray or a list of these.')
            elif isinstance(self.params[stringorlist_list[i]], (str,np.ndarray)):
                self.params[stringorlist_list[i]] = [self.params[stringorlist_list[i]]]
        
        for i in range(len(numberorlist_list)):
            if not isinstance(self.params[numberorlist_list[i]], (int,float)) and not isinstance(self.params[numberorlist_list[i]], list):
                flag = -1
                print('ERROR: ' + numberorlist_list[i] + ' must be a number or a list of numbers.')
            elif isinstance(self.params[numberorlist_list[i]], (int,float)):
                self.params[numberorlist_list[i]] = [self.params[numberorlist_list[i]]]
        
        for i in range(len(noneorarrayorlist_list)):
            if self.params[noneorarrayorlist_list[i]] is not None and not isinstance(self.params[noneorarrayorlist_list[i]], list) and not isinstance(self.params[noneorarrayorlist_list[i]], np.ndarray):
                flag = -1
                print('ERROR: ' + noneorarrayorlist_list[i] + ' must be None or a list or a np.ndarray.')
            elif isinstance(self.params[noneorarrayorlist_list[i]], str):
                self.params[noneorarrayorlist_list[i]] = [self.params[noneorarrayorlist_list[i]]]
            elif isinstance(self.params[noneorarrayorlist_list[i]], np.ndarray):
                self.params[noneorarrayorlist_list[i]] = np.ndarray.tolist(self.params[noneorarrayorlist_list[i]])
        
        # check channel length
        channel_check_list = ['sigma','template','target']
        channels = [len(self.params[x]) for x in channel_check_list]
        channel_set = list(set(channels))
        if len(channel_set) > 2 or (len(channel_set) == 2 and 1 not in channel_set):
            print('ERROR: number of channels is not the same between sigma, template, and target.')
            flag = -1
        elif len(self.params['template']) != len(self.params['target']):
            print('ERROR: number of channels is not the same between template and target.')
            flag = -1
        elif len(self.params['sigma']) > 1 and len(self.params['sigma']) != len(self.params['template']):
            print('ERROR: sigma does not have channels of size 1 or # of template channels.')
            flag = -1
        elif (len(channel_set) == 2 and 1 in channel_set):
            channel_set.remove(1)
            for i in range(len(channel_check_list)):
                if channels[i] == 1:
                    self.params[channel_check_list[i]] = self.params[channel_check_list[i]]*channel_set[0]
            
            print('WARNING: one or more of sigma, template, and target has length 1 while another does not.')
        
        # check contrast correction channels
        if isinstance(self.params['cc_channels'],(int,float)):
            self.params['cc_channels'] = [int(self.params['cc_channels'])]
        elif isinstance(self.params['cc_channels'],list):
            if len(self.params['cc_channels']) == 0:
                self.params['cc_channels'] = list(range(max(channel_set)))
            
            if max(self.params['cc_channels']) > max(channel_set):
                print('ERROR: one or more of the contrast correction channels is greater than the number of image channels.')
                flag = -1
        
        # check weight estimation channels
        if isinstance(self.params['we_channels'],(int,float)):
            self.params['we_channels'] = [int(self.params['we_channels'])]
        elif isinstance(self.params['we_channels'],list):
            if len(self.params['we_channels']) == 0:
                self.params['we_channels'] = list(range(max(channel_set)))
            
            if max(self.params['we_channels']) > max(channel_set):
                print('ERROR: one or more of the weight estimation channels is greater than the number of image channels.')
                flag = -1
        
        # check weight estimation sigmas
        if len(self.params['sigmaW']) == 1:
            self.params['sigmaW'] = self.params['sigmaW']*int(self.params['we'])
        elif len(self.params['sigmaW']) != int(self.params['we']):
            print('ERROR: length of weight estimation sigma list must be either 1 or equal to parameter \'we\'.')
            flag = -1
        
        # optimizer flags
        if self.params['optimizer'] == 'gdw':
            self.params['savebestv'] = True
        
        # set timesteps to 1 if doing affine only
        if self.params['do_affine'] > 0 and self.params['do_lddmm'] == 0 and not hasattr(self,'vt0'):
            if self.params['nt'] != 1:
                print('WARNING: nt set to 1 because settings indicate affine registration only.')
                self.params['nt'] = 1
        elif self.params['do_affine'] == 0 and self.params['do_lddmm'] == 0:
            flag = -1
            print('ERROR: both linear and LDDMM registration are turned off. Exiting.')
        elif self.params['do_lddmm'] == 1:
            if self.params['nt'] == 1 and self.params['orig_nt'] == 1:
                print('WARNING: parameter \'nt\' is currently set to 1. You might have just finished linear registration. For LDDMM, set to a higher value.')
            elif self.params['nt'] == 1 and self.params['orig_nt'] != 1:
                self.params['nt'] = self.params['orig_nt']
                print('WARNING: parameter \'nt\' was set to 1 and has been automatically reverted to your initial value of ' + str(self.params['orig_nt']) + '.')
        
        return flag
    
    # helper function to load images
    def _load(self, template, target, costmask):
        if isinstance(template, str):
            I = [None]
            Ispacing = [None]
            Isize = [None]
            I[0],Ispacing[0],Isize[0] = self.loadImage(template,im_norm_ms=self.params['im_norm_ms'])
        elif isinstance(template, np.ndarray):
            I = [None]
            Ispacing = [None]
            Isize = [None]
            if self.params['im_norm_ms'] == 1:
                I[0] = torch.tensor((template - np.mean(template)) / np.std(template)).type(self.params['dtype']).to(device=self.params['cuda'])
            else:
                I[0] = torch.tensor(template).type(self.params['dtype']).to(device=self.params['cuda'])
            
            Isize[0] = list(template.shape)
            if self.params['dx'] == None:
                Ispacing[0] = np.ones((3,)).astype(np.float32)
            else:
                Ispacing[0] = self.params['dx']
        elif isinstance(template, list):
            if isinstance(template[0],str):
                I = [None]*len(template)
                Ispacing = [None]*len(template)
                Isize = [None]*len(template)
                for i in range(len(template)):
                    I[i],Ispacing[i],Isize[i] = self.loadImage(template[i],im_norm_ms=self.params['im_norm_ms'])
            # assumes images are the same spacing
            elif isinstance(template[0],np.ndarray):
                I = [None]*len(template)
                Ispacing = [None]*len(template)
                Isize = [None]*len(template)
                for i in range(len(template)):
                    if self.params['im_norm_ms'] == 1:
                        I[i] = torch.tensor((template[i] - np.mean(template[i])) / np.std(template[i])).type(self.params['dtype']).to(device=self.params['cuda'])
                    else:
                        I[i] = torch.tensor(template[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    
                    Isize[i] = template[i].shape
                    if self.params['dx'] == None:
                        Ispacing[i] = np.ones((3,)).astype(np.float32)
                    else:
                        Ispacing[i] = self.params['dx']
            else:
                print('ERROR: received list of unhandled type for template image.')
                return -1
        
        if isinstance(target, str):
            J = [None]
            Jspacing = [None]
            Jsize = [None]
            J[0],Jspacing[0],Jsize[0] = self.loadImage(target,im_norm_ms=self.params['im_norm_ms'])
        elif isinstance(target, np.ndarray):
            J = [None]
            Jspacing = [None]
            Jsize = [None]
            if self.params['im_norm_ms'] == 1:
                J[0] = torch.tensor((target - np.mean(target)) / np.std(target)).type(self.params['dtype']).to(device=self.params['cuda'])
            else:
                J[0] = torch.tensor(target).type(self.params['dtype']).to(device=self.params['cuda'])
            
            Jsize[0] = list(target.shape)
            if self.params['dx'] == None:
                Jspacing[0] = np.ones((3,)).astype(np.float32)
            else:
                Jspacing[0] = self.params['dx']
        elif isinstance(target, list):
            if isinstance(target[0],str):
                J = [None]*len(target)
                Jspacing = [None]*len(target)
                Jsize = [None]*len(target)
                for i in range(len(target)):
                    J[i],Jspacing[i],Jsize[i] = self.loadImage(target[i],im_norm_ms=self.params['im_norm_ms'])
            # assumes images are the same spacing
            elif isinstance(target[0],np.ndarray):
                J = [None]*len(target)
                Jspacing = [None]*len(target)
                Jsize = [None]*len(target)
                for i in range(len(target)):
                    if self.params['im_norm_ms'] == 1:
                        J[i] = torch.tensor((target[i] - np.mean(target[i])) / np.std(target[i])).type(self.params['dtype']).to(device=self.params['cuda'])
                    else:
                        J[i] = torch.tensor(target[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    
                    Jsize[i] = target[i].shape
                    if self.params['dx'] == None:
                        Jspacing[i] = np.ones((3,)).astype(np.float32)
                    else:
                        Jspacing[i] = self.params['dx']
            else:
                print('ERROR: received list of unhandled type for target image.')
                return -1
        
        # load costmask if the variable exists
        # TODO: make this multichannel
        if isinstance(costmask, str):
            K = [None]
            Kspacing = [None]
            Ksize = [None]
            # never normalize cost mask
            K[0],Kspacing[0],Ksize[0] = self.loadImage(costmask,im_norm_ms=0)
        elif isinstance(costmask,np.ndarray):
            K = [None]
            Kspacing = [None]
            Ksize = [None]
            K[0] = torch.tensor(costmask).type(self.params['dtype']).to(device=self.params['cuda'])
            Ksize[0] = costmask.shape
            if self.params['dx'] == None:
                Kspacing[0] = np.ones((3,)).astype(np.float32)
            else:
                Kspacing[0] = self.params['dx']
        else:
            K = []
            Kspacing = []
            Ksize = []
        
        if len(J) != len(I):
            print('ERROR: images must have the same number of channels.')
            return -1
            
        #if I.shape[0] != J.shape[0] or I.shape[1] != J.shape[1] or I.shape[2] != J.shape[2]:
        #if I.shape != J.shape:
        if not all([x.shape == I[0].shape for x in I+J+K]):
            print('ERROR: the image sizes are not the same.\n')
            return -1
        #elif Ispacing[0] != Jspacing[0] or Ispacing[1] != Jspacing[1] or Ispacing[2] != Jspacing[2]
        #elif np.sum(Ispacing==Jspacing) < len(I.shape):
        elif self.params['dx'] is None and not all([list(x == Ispacing[0]) for x in Ispacing+Jspacing+Kspacing]):
            print('ERROR: the image pixel spacings are not the same.\n')
            return -1
        else:
            self.I = I
            self.J = J
            if costmask is not None:
                self.M = K[0]
            else:
                self.M = torch.tensor(np.ones(I[0].shape)).type(self.params['dtype']).to(device=self.params['cuda']) # this could be initialized to a scalar 1.0 to save memory, if you do this make sure you check computeLinearContrastCorrection to see whether or not you are using torch.sum(w*self.M)
            
            self.dx = list(Ispacing[0])
            self.dx = [float(x) for x in self.dx]
            self.nx = I[0].shape
            return 1
    
    # initialize lddmm kernels
    def initializeKernels(self):
        # make smoothing kernel on CPU
        f0 = np.linspace(0,self.nx[0]-1,int(np.round(self.nx[0]*self.params['v_scale'])))/(self.dx[0]*self.nx[0])
        f1 = np.linspace(0,self.nx[1]-1,int(np.round(self.nx[1]*self.params['v_scale'])))/(self.dx[1]*self.nx[1])
        f2 = np.linspace(0,self.nx[2]-1,int(np.round(self.nx[2]*self.params['v_scale'])))/(self.dx[2]*self.nx[2])
        F0,F1,F2 = np.meshgrid(f0,f1,f2,indexing='ij')
        #a = 3.0*self.dx[0] # a scale in mm
        #p = 2
        self.Ahat = (1.0 - 2.0*(self.params['a']*self.dx[0])**2*((np.cos(2.0*np.pi*self.dx[0]*F0) - 1.0)/self.dx[0]**2 
                                + (np.cos(2.0*np.pi*self.dx[1]*F1) - 1.0)/self.dx[1]**2
                                + (np.cos(2.0*np.pi*self.dx[2]*F2) - 1.0)/self.dx[2]**2))**(2.0*self.params['p'])
        self.Khat = 1.0/self.Ahat
        # only move one kernel for now
        # TODO: try broadcasting this instead
        self.Khat = torch.tensor(np.tile(np.reshape(self.Khat,(self.Khat.shape[0],self.Khat.shape[1],self.Khat.shape[2],1)),(1,1,1,2))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # optimization multipliers (putting this in here because I want to reset this if I change the smoothing kernel)
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        #self.GDBetaAffineR = float(1.0)
        #self.GDBetaAffineT = float(1.0)
        self.climbcount = 0
        if self.params['savebestv']:
            self.best = {}
    
    # initialize lddmm kernels
    def initializeKernels2d(self):
        # make smoothing kernel on CPU
        f0 = np.arange(self.nx[0])/(self.dx[0]*self.nx[0])
        f1 = np.arange(self.nx[1])/(self.dx[1]*self.nx[1])
        F0,F1 = np.meshgrid(f0,f1,indexing='ij')
        #a = 3.0*self.dx[0] # a scale in mm
        #p = 2
        self.Ahat = (1.0 - 2.0*(self.params['a']*self.dx[0])**2*((np.cos(2.0*np.pi*self.dx[0]*F0) - 1.0)/self.dx[0]**2 
                                + (np.cos(2.0*np.pi*self.dx[1]*F1) - 1.0)/self.dx[1]**2))**(2.0*self.params['p'])
        self.Khat = 1.0/self.Ahat
        # only move one kernel for now
        # TODO: try broadcasting this instead
        self.Khat = torch.tensor(np.tile(np.reshape(self.Khat,(self.Khat.shape[0],self.Khat.shape[1],1)),(1,1,2))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # optimization multipliers (putting this in here because I want to reset this if I change the smoothing kernel)
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        #self.GDBetaAffineR = float(1.0)
        #self.GDBetaAffineT = float(1.0)
        self.climbcount = 0
        if self.params['savebestv']:
            self.best = {}
    
    # initialize lddmm variables
    def initializeVariables(self):
        # TODO: handle 2D and 3D versions
        # helper variables
        self.dt = 1.0/self.params['nt']
        # loss values
        if not hasattr(self,'EMAll'):
            self.EMAll = []
        if not hasattr(self,'ERAll'):
            self.ERAll = []
        if not hasattr(self,'EAll'):
            self.EAll = []
        if self.params['checkaffinestep'] == 1:
            if not hasattr(self,'EMAffineR'):
                self.EMAffineR = []
            if not hasattr(self,'EMAffineT'):
                self.EMAffineT = []
            if not hasattr(self,'EMDiffeo'):
                self.EMDiffeo = []
        
        # save X if v_scale changed
        #if hasattr(self, 'vt0') and self.initializer_flags['v_scale'] == 1:
        #    old_X0 = self.X0.clone()
        #    old_X1 = self.X0.clone()
        #    old_X2 = self.X0.clone()
        
        # image sampling domain
        x0 = np.linspace(0,self.nx[0]-1,int(np.round(self.nx[0]*self.params['v_scale'])))*self.dx[0]
        x1 = np.linspace(0,self.nx[1]-1,int(np.round(self.nx[1]*self.params['v_scale'])))*self.dx[1]
        x2 = np.linspace(0,self.nx[2]-1,int(np.round(self.nx[2]*self.params['v_scale'])))*self.dx[2]
        #x0 = np.arange(self.nx[0])*self.dx[0]
        #x1 = np.arange(self.nx[1])*self.dx[1]
        #x2 = np.arange(self.nx[2])*self.dx[2]
        X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')
        self.X0 = torch.tensor(X0-np.mean(X0)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.X1 = torch.tensor(X1-np.mean(X1)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.X2 = torch.tensor(X2-np.mean(X2)).type(self.params['dtype']).to(device=self.params['cuda'])
        # 2D sampling domain for slice alignment
        #if self.params['slice_alignment'] == 1:
        
        # load a gaussian filter if v_scale is less than 1
        if self.params['v_scale'] < 1.0:
            size = int(np.ceil(1.0/self.params['v_scale']*5))
            if np.mod(size,2) == 0:
                size += 1
            
            self.gaussian_filter = torch.tensor(mygaussian3d(sigma=1.0/self.params['v_scale'],size=size)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # v and I
        if self.params['gpu_number'] is not None:
            if not hasattr(self, 'vt0') and self.initializer_flags['lddmm'] == 1: # we never reset lddmm variables
                self.vt0 = []
                self.vt1 = []
                self.vt2 = []
                for i in range(self.params['nt']):
                    self.vt0.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                    self.vt1.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                    self.vt2.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
            
            if (self.initializer_flags['load'] == 1 or self.initializer_flags['lddmm'] == 1) and self.params['low_memory'] < 1:
                self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
                for ii in range(len(self.I)):
                    # NOTE: you cannot use pointers / list multiplication for cuda tensors if you want actual copies
                    #self.It.append(torch.tensor(self.I[:,:,:]).type(self.params['dtype']).cuda())
                    for i in range(self.params['nt']+1):
                        if i == 0:
                            self.It[ii][i] = self.I[ii]
                        else:
                            if isinstance(self.I[ii],torch.Tensor):
                                self.It[ii][i] = self.I[ii][:,:,:].clone().type(self.params['dtype']).cuda()
                            else:
                                self.It[ii][i] = torch.tensor(self.I[ii][:,:,:]).type(self.params['dtype']).cuda()
        else:
            if not hasattr(self,'vt0') and self.initializer_flags['lddmm'] == 1:
                self.vt0 = []
                self.vt1 = []
                self.vt2 = []
                for i in range(self.params['nt']):
                    self.vt0.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']))
                    self.vt1.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']))
                    self.vt2.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']))
            
            #self.It = [[None]]*len(self.I)
            #for i in range(len(self.I)):
            #    self.It[i] = [torch.tensor(self.I[i][:,:,:]).type(self.params['dtype'])]*(self.params['nt']+1)
            if self.initializer_flags['load'] == 1 or self.initializer_flags['lddmm'] == 1:
                self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
                for ii in range(len(self.I)):
                    # NOTE: you cannot use pointers / list multiplication for cuda tensors if you want actual copies
                    #self.It.append(torch.tensor(self.I[:,:,:]).type(self.params['dtype']).cuda())
                    #for i in range(self.params['nt']+1):
                    #    self.It[ii][i] = torch.tensor(self.I[ii][:,:,:]).type(self.params['dtype'])
                    for i in range(self.params['nt']+1):
                        if i == 0:
                            self.It[ii][i] = self.I[ii]
                        else:
                            if isinstance(self.I[ii],torch.Tensor):
                                self.It[ii][i] = self.I[ii][:,:,:].clone().type(self.params['dtype'])
                            else:
                                self.It[ii][i] = torch.tensor(self.I[ii][:,:,:]).type(self.params['dtype'])
        
        # check if v_scale has changed
        if hasattr(self, 'vt0') and self.initializer_flags['v_scale'] == 1:
            # resample the current v fields
            for i in range(self.params['nt']):
                self.vt0[i] = torch.squeeze(grid_sample((self.vt0[i]).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border'))
                self.vt1[i] = torch.squeeze(grid_sample((self.vt1[i]).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border'))
                self.vt2[i] = torch.squeeze(grid_sample((self.vt2[i]).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border'))
        
        # affine parameters
        if not hasattr(self,'affineA') and self.initializer_flags['affine'] == 1: # we never automatically reset affine variables
            self.affineA = torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.lastaffineA = torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.gradA = torch.tensor(np.zeros((4,4))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # contrast correction variables
        if not hasattr(self,'ccIbar') or self.initializer_flags['cc'] == 1: # we never reset cc variables
            self.ccIbar = []
            self.ccJbar = []
            self.ccVarI = []
            self.ccCovIJ = []
            for i in range(len(self.I)):
                self.ccIbar.append(0.0)
                self.ccJbar.append(0.0)
                self.ccVarI.append(1.0)
                self.ccCovIJ.append(1.0)
        
        # contrast correction variables
        if not hasattr(self,'ccCoeff') or self.initializer_flags['cc'] == 1: # we never reset cc variables
            self.ccIbar = []
            self.ccJbar = []
            self.ccVarI = []
            self.ccCovIJ = []
            for i in range(len(self.I)):
                self.ccIbar.append(0.0)
                self.ccJbar.append(0.0)
                self.ccVarI.append(1.0)
                self.ccCovIJ.append(1.0)
        
        # weight estimation variables
        if self.initializer_flags['we'] == 1: # if number of channels changed, reset everything
            self.W = [[] for i in range(len(self.I))]
            self.we_C = [[] for i in range(len(self.I))]
            for i in range(self.params['we']):
                if i == 0: # first index is the matching channel, the rest is artifacts
                    for ii in self.params['we_channels']: # allocate space only for the desired channels
                        self.W[ii].append(torch.tensor(0.9*np.ones((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
                else:
                    for ii in self.params['we_channels']:
                        self.W[ii].append(torch.tensor(0.1*np.ones((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # optimizer update variables
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.GDBetaAffineR = float(1.0)
        self.GDBetaAffineT = float(1.0)
        
        # adam optimizer variables
        if self.params['optimizer'] == "adam":
            self.sgd_M = torch.ones(self.M.shape).type(self.params['dtype']).to(device=self.params['cuda'])
            self.sgd_maskiter = 0
            #self.params['adam_alpha'] = torch.Tensor(self.params['adam_alpha']).type(self.params['dtype']).to(device=self.params['cuda'])
            #self.params['adam_beta1'] = torch.Tensor(self.params['adam_beta1']).type(self.params['dtype']).to(device=self.params['cuda'])
            #self.params['adam_beta2'] = torch.Tensor(self.params['adam_beta2']).type(self.params['dtype']).to(device=self.params['cuda'])
            #self.params['adam_epsilon'] = torch.Tensor(self.params['adam_epsilon']).type(self.params['dtype']).to(device=self.params['cuda'])
            self.adam = {}
            self.adam['m0'] = []
            self.adam['m1'] = []
            self.adam['m2'] = []
            self.adam['v0'] = []
            self.adam['v1'] = []
            self.adam['v2'] = []
            for i in range(self.params['nt']):
                self.adam['m0'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['m1'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['m2'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['v0'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['v1'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adam['v2'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # adadelta optimizer variables
        if self.params['optimizer'] == 'adadelta':
            self.adadelta = {}
            # here we call m0 the accumulator for the gradients and v0 the accumulator for the parameter itself
            self.adadelta['m0'] = []
            self.adadelta['m1'] = []
            self.adadelta['m2'] = []
            self.adadelta['v0'] = []
            self.adadelta['v1'] = []
            self.adadelta['v2'] = []
            for i in range(self.params['nt']):
                self.adadelta['m0'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adadelta['m1'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adadelta['m2'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adadelta['v0'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adadelta['v1'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.adadelta['v2'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # rmsprop optimizer variables
        if self.params['optimizer'] == 'rmsprop':
            self.sgd_M = torch.ones(self.M.shape).type(self.params['dtype']).to(device=self.params['cuda'])
            self.sgd_maskiter = 0
            self.rmsprop = {}
            # here we call m0 the accumulator for the gradients and v0 the accumulator for the parameter itself
            self.rmsprop['m0'] = []
            self.rmsprop['m1'] = []
            self.rmsprop['m2'] = []
            for i in range(self.params['nt']):
                self.rmsprop['m0'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.rmsprop['m1'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.rmsprop['m2'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # SGD mask initialization
        if self.params['optimizer'] == 'sgd':
            self.sgd_M = torch.ones(self.M.shape).type(self.params['dtype']).to(device=self.params['cuda'])
            self.sgd_maskiter = 0
        
        # SGDM initialization
        if self.params['optimizer'] == 'sgdm':
            self.sgd_M = torch.ones(self.M.shape).type(self.params['dtype']).to(device=self.params['cuda'])
            # here we call m0 the accumulator for the gradients and v0 the accumulator for the parameter itself
            self.sgdm = {}
            self.sgdm['m0'] = []
            self.sgdm['m1'] = []
            self.sgdm['m2'] = []
            for i in range(self.params['nt']):
                self.sgdm['m0'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.sgdm['m1'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.sgdm['m2'].append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale'])),int(np.round(self.nx[2]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
            self.sgd_maskiter = 0
        
        # reset initializer flags
        self.initializer_flags['load'] = 0
        self.initializer_flags['lddmm'] = 0
        self.initializer_flags['affine'] = 0
        self.initializer_flags['cc'] = 0
        self.initializer_flags['we'] = 0
        self.initializer_flags['v_scale'] = 0
    
    
    # initialize lddmm variables
    def initializeVariables2d(self):
        # TODO: handle 2D and 3D versions
        # helper variables
        self.dt = 1.0/self.params['nt']
        # loss values
        if not hasattr(self,'EMAll'):
            self.EMAll = []
        if not hasattr(self,'ERAll'):
            self.ERAll = []
        if not hasattr(self,'EAll'):
            self.EAll = []
        if self.params['checkaffinestep'] == 1:
            if not hasattr(self,'EMAffineR'):
                self.EMAffineR = []
            if not hasattr(self,'EMAffineT'):
                self.EMAffineT = []
            if not hasattr(self,'EMDiffeo'):
                self.EMDiffeo = []
        
        # load a gaussian filter if v_scale is less than 1
        if self.params['v_scale'] < 1.0:
            size = int(np.ceil(1.0/self.params['v_scale']*5))
            if np.mod(size,2) == 0:
                size += 1
            
            self.gaussian_filter = torch.tensor(mygaussian(sigma=1.0/self.params['v_scale'],size=size)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # image sampling domain
        x0 = np.arange(self.nx[0])*self.dx[0]
        x1 = np.arange(self.nx[1])*self.dx[1]
        X0,X1 = np.meshgrid(x0,x1,indexing='ij')
        self.X0 = torch.tensor(X0-np.mean(X0)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.X1 = torch.tensor(X1-np.mean(X1)).type(self.params['dtype']).to(device=self.params['cuda'])
        '''
        # v and I
        if self.params['gpu_number'] is not None:
            self.vt0 = []
            self.vt1 = []
            self.detjac = []
            self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
            for ii in range(len(self.I)):
                for i in range(self.params['nt']+1):
                    self.It[ii][i] = torch.tensor(self.I[ii][:,:]).type(self.params['dtype']).cuda()
            
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
                self.detjac.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
        else:
            self.vt0 = []
            self.vt1 = []
            self.detjac = []
            self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
            for ii in range(len(self.I)):
                for i in range(self.params['nt']+1):
                    self.It[ii][i] = torch.tensor(self.I[ii][:,:]).type(self.params['dtype'])
            
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']))
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']))
                self.detjac.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']))
        '''
        # v and I
        if self.params['gpu_number'] is not None:
            if not hasattr(self, 'vt0') and self.initializer_flags['lddmm'] == 1: # we never reset lddmm variables
                self.vt0 = []
                self.vt1 = []
                self.detjac = []
                for i in range(self.params['nt']):
                    self.vt0.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                    self.vt1.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
                    self.detjac.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']).to(device=self.params['cuda']))
            
            if (self.initializer_flags['load'] == 1 or self.initializer_flags['lddmm'] == 1) and self.params['low_memory'] < 1:
                self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
                for ii in range(len(self.I)):
                    # NOTE: you cannot use pointers / list multiplication for cuda tensors if you want actual copies
                    #self.It.append(torch.tensor(self.I[:,:,:]).type(self.params['dtype']).cuda())
                    for i in range(self.params['nt']+1):
                        if i == 0:
                            self.It[ii][i] = self.I[ii]
                        else:
                            if isinstance(self.I[ii],torch.Tensor):
                                self.It[ii][i] = self.I[ii][:,:].clone().type(self.params['dtype']).cuda()
                            else:
                                self.It[ii][i] = torch.tensor(self.I[ii][:,:]).type(self.params['dtype']).cuda()
        else:
            if not hasattr(self,'vt0') and self.initializer_flags['lddmm'] == 1:
                self.vt0 = []
                self.vt1 = []
                self.detjac = []
                for i in range(self.params['nt']):
                    self.vt0.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']))
                    self.vt1.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']))
                    self.detjac.append(torch.tensor(np.zeros((int(np.round(self.nx[0]*self.params['v_scale'])),int(np.round(self.nx[1]*self.params['v_scale']))))).type(self.params['dtype']))
            
            #self.It = [[None]]*len(self.I)
            #for i in range(len(self.I)):
            #    self.It[i] = [torch.tensor(self.I[i][:,:,:]).type(self.params['dtype'])]*(self.params['nt']+1)
            if self.initializer_flags['load'] == 1 or self.initializer_flags['lddmm'] == 1:
                self.It = [ [None]*(self.params['nt']+1) for i in range(len(self.I)) ]
                for ii in range(len(self.I)):
                    # NOTE: you cannot use pointers / list multiplication for cuda tensors if you want actual copies
                    #self.It.append(torch.tensor(self.I[:,:,:]).type(self.params['dtype']).cuda())
                    for i in range(self.params['nt']+1):
                        self.It[ii][i] = torch.tensor(self.I[ii][:,:]).type(self.params['dtype'])
        
        # affine parameters
        if not hasattr(self,'affineA') and self.initializer_flags['affine'] == 1: # we never automatically reset affine variables
            self.affineA = torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.lastaffineA = torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.gradA = torch.tensor(np.zeros((3,3))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # optimizer update variables
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.GDBetaAffineR = float(1.0)
        self.GDBetaAffineT = float(1.0)
        
        # contrast correction variables
        if not hasattr(self,'ccIbar') or self.initializer_flags['cc'] == 1: # we never reset cc variables
            self.ccIbar = []
            self.ccJbar = []
            self.ccVarI = []
            self.ccCovIJ = []
            for i in range(len(self.I)):
                self.ccIbar.append(0.0)
                self.ccJbar.append(0.0)
                self.ccVarI.append(1.0)
                self.ccCovIJ.append(1.0)
        
        # weight estimation variables
        if self.initializer_flags['we'] == 1: # if number of channels changed, reset everything
            self.W = [[] for i in range(len(self.I))]
            self.we_C = [[] for i in range(len(self.I))]
            for i in range(self.params['we']):
                if i == 0: # first index is the matching channel, the rest is artifacts
                    for ii in self.params['we_channels']: # allocate space only for the desired channels
                        self.W[ii].append(torch.tensor(0.9*np.ones((self.nx[0],self.nx[1]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
                else:
                    for ii in self.params['we_channels']:
                        self.W[ii].append(torch.tensor(0.1*np.ones((self.nx[0],self.nx[1]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # reset initializer flags
        self.initializer_flags['load'] = 0
        self.initializer_flags['lddmm'] = 0
        self.initializer_flags['affine'] = 0
        self.initializer_flags['cc'] = 0
        self.initializer_flags['we'] = 0
        self.initializer_flags['v_scale'] = 0
    
    # helper function for torch_gradient
    def _allocateGradientDivisors(self):
        if self.J[0].dim() == 3:
            # allocate gradient divisor for custom torch gradient function
            self.grad_divisor_x = np.ones(self.I[0].shape)
            self.grad_divisor_x[1:-1,:,:] = 2
            self.grad_divisor_x = torch.tensor(self.grad_divisor_x).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_y = np.ones(self.I[0].shape)
            self.grad_divisor_y[:,1:-1,:] = 2
            self.grad_divisor_y = torch.tensor(self.grad_divisor_y).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_z = np.ones(self.I[0].shape)
            self.grad_divisor_z[:,:,1:-1] = 2
            self.grad_divisor_z = torch.tensor(self.grad_divisor_z).type(self.params['dtype']).to(device=self.params['cuda'])
        else:
            # allocate gradient divisor for custom torch gradient function
            self.grad_divisor_x = np.ones(self.I[0].shape)
            self.grad_divisor_x[1:-1,:] = 2
            self.grad_divisor_x = torch.tensor(self.grad_divisor_x).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_y = np.ones(self.I[0].shape)
            self.grad_divisor_y[:,1:-1] = 2
            self.grad_divisor_y = torch.tensor(self.grad_divisor_y).type(self.params['dtype']).to(device=self.params['cuda'])
    
    # replication-pad, artificial roll, subtract, single-sided difference on boundaries
    def torch_gradient(self,arr, dx, dy, dz, grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu):
        arr = torch.squeeze(torch.nn.functional.pad(arr.unsqueeze(0).unsqueeze(0),(1,1,1,1,1,1),mode='replicate'))
        gradx = torch.cat((arr[1:,:,:],arr[0,:,:].unsqueeze(0)),dim=0) - torch.cat((arr[-1,:,:].unsqueeze(0),arr[:-1,:,:]),dim=0)
        grady = torch.cat((arr[:,1:,:],arr[:,0,:].unsqueeze(1)),dim=1) - torch.cat((arr[:,-1,:].unsqueeze(1),arr[:,:-1,:]),dim=1)
        gradz = torch.cat((arr[:,:,1:],arr[:,:,0].unsqueeze(2)),dim=2) - torch.cat((arr[:,:,-1].unsqueeze(2),arr[:,:,:-1]),dim=2)
        return gradx[1:-1,1:-1,1:-1]/dx/grad_divisor_x_gpu, grady[1:-1,1:-1,1:-1]/dy/grad_divisor_y_gpu, gradz[1:-1,1:-1,1:-1]/dz/grad_divisor_z_gpu
    
    # 2D replication-pad, artificial roll, subtract, single-sided difference on boundaries
    def torch_gradient2d(self,arr, dx, dy, grad_divisor_x_gpu,grad_divisor_y_gpu):
        arr = torch.squeeze(torch.nn.functional.pad(arr.unsqueeze(0).unsqueeze(0),(1,1,1,1),mode='replicate'))
        gradx = torch.cat((arr[1:,:],arr[0,:].unsqueeze(0)),dim=0) - torch.cat((arr[-1,:].unsqueeze(0),arr[:-1,:]),dim=0)
        grady = torch.cat((arr[:,1:],arr[:,0].unsqueeze(1)),dim=1) - torch.cat((arr[:,-1].unsqueeze(1),arr[:,:-1]),dim=1)
        return gradx[1:-1,1:-1]/dx/grad_divisor_x_gpu, grady[1:-1,1:-1]/dy/grad_divisor_y_gpu
    
    # deform template forward
    def forwardDeformation(self):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        phiinv2_gpu = self.X2.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self, 'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
                phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X2-self.vt2[t]*self.dt)
            
            # do affine transforms
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0 or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                if self.params['checkaffinestep'] == 1:
                    # new diffeo with old affine
                    # this doesn't match up with EAll even when vt is identity
                    phiinv0_temp,phiinv1_temp,phiinv2_temp = self.forwardDeformationAffineVectorized(self.lastaffineA.clone(),phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    I = [None]*len(self.I)
                    for i in range(len(self.I)):
                        if self.params['v_scale'] != 1.0:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv2_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[2]*self.dx[2]-self.dx[2])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv1_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))
                        else:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_temp/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))
                    
                    self.EMDiffeo.append( self.calculateMatchingEnergyMSEOnly(I) )
                    # new diffeo with new L and old T
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineR(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    phiinv0_temp,phiinv1_temp,phiinv2_temp = self.forwardDeformationAffineT(self.lastaffineA.clone(),phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    I = [None]*len(self.I)
                    for i in range(len(self.I)):
                        if self.params['v_scale'] != 1.0:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv2_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[2]*self.dx[2]-self.dx[2])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv1_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))
                        else:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_temp/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))
                    
                    self.EMAffineR.append( self.calculateMatchingEnergyMSEOnly(I) )
                    # new everything
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineT(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    del phiinv0_temp,phiinv1_temp,phiinv2_temp
                else:
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineVectorized(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
            
            # deform the image
            for i in range(len(self.I)):
                if self.params['v_scale'] != 1.0:
                    self.It[i][t+1] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv2_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[2]*self.dx[2]-self.dx[2])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))
                else:
                    self.It[i][t+1] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))
        
        del phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
        #return self.It,phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # compute current displacement field
    def computeThisDisplacement(self,interpmode='bilinear'):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        phiinv2_gpu = self.X2.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
                phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X2-self.vt2[t]*self.dt)
            
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineVectorized(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
        
        phiinv0_gpu -= self.X0
        phiinv1_gpu -= self.X1
        phiinv2_gpu -= self.X2
        return phiinv0_gpu.cpu().numpy(),phiinv1_gpu.cpu().numpy(),phiinv2_gpu.cpu().numpy()

    # apply current transform to new image
    def applyThisTransform(self, I, interpmode='bilinear',dtype='torch.FloatTensor'):
        It = []
        for i in range(self.params['nt']+1):
            It.append(torch.tensor(I).type(dtype).to(device=self.params['cuda']))
        
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        phiinv2_gpu = self.X2.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
                phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X2-self.vt2[t]*self.dt)
            
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineVectorized(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
            
            # deform the image
            if self.params['v_scale'] != 1.0:
                It[t+1] = torch.squeeze(grid_sample(It[0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv2_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[2]*self.dx[2]-self.dx[2])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros',mode=interpmode))
            else:
                It[t+1] = torch.squeeze(grid_sample(It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        
        return It,phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # apply current transform to new image
    def applyThisTransform2d(self, I, interpmode='bilinear',dtype='torch.FloatTensor'):
        It = []
        for i in range(self.params['nt']+1):
            It.append(torch.tensor(I).type(dtype).to(device=self.params['cuda']))
        
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
            
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineVectorized2d(self.affineA,phiinv0_gpu,phiinv1_gpu)
            
            # deform the image
            if self.params['v_scale'] != 1.0:
                It[t+1] = torch.squeeze(grid_sample(It[0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros',mode=interpmode))
            else:
                It[t+1] = torch.squeeze(grid_sample(It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        
        return It,phiinv0_gpu, phiinv1_gpu
    
    # apply current transform to new image
    def applyThisTransformNT(self, I, interpmode='bilinear',dtype='torch.FloatTensor',nt=None):
        if self.J[0].dim() == 2:
            It = self.applyThisTransformNT2d(I, interpmode='bilinear',dtype='torch.FloatTensor',nt=None)
        elif self.J[0].dim() == 3:
            It = self.applyThisTransformNT3d(I, interpmode='bilinear',dtype='torch.FloatTensor',nt=None)
        
        return It
    
    # apply current transform to new image
    def applyThisTransformNT3d(self, I, interpmode='bilinear',dtype='torch.FloatTensor',nt=None):
        if nt == None:
            nt = self.params['nt']
        
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        phiinv2_gpu = self.X2.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(nt):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
                phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X2-self.vt2[t]*self.dt)
            
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineVectorized(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
        
        # deform the image
        # TODO: do I actually need to send phiinv to gpu here?
        if self.params['v_scale'] != 1.0:
            It = torch.squeeze(grid_sample(I.unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv2_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[2]*self.dx[2]-self.dx[2])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1],self.nx[2]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        else:
            It = torch.squeeze(grid_sample(I.unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        
        del phiinv0_gpu,phiinv1_gpu,phiinv2_gpu
        return It
    
    # apply current transform to new image
    def applyThisTransformNT2d(self, I, interpmode='bilinear',dtype='torch.FloatTensor',nt=None):
        if nt == None:
            nt = self.params['nt']
        
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(nt):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self,'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
            
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0  or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineVectorized(self.affineA,phiinv0_gpu,phiinv1_gpu)
        
        # deform the image
        # TODO: do I actually need to send phiinv to gpu here?
        if self.params['v_scale'] != 1.0:
            It = torch.squeeze(grid_sample(I.unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='trilinear',align_corners=True)).type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        else:
            It = torch.squeeze(grid_sample(I.unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu.type(dtype).to(device=self.params['cuda'])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros',mode=interpmode))
        
        del phiinv0_gpu,phiinv1_gpu
        return It
    
    # deform template forward
    def forwardDeformation2d(self):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            if self.params['do_lddmm'] == 1 or hasattr(self, 'vt0'):
                phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt) 
            
            # do affine transforms
            if t == self.params['nt']-1 and (self.params['do_affine'] > 0 or (hasattr(self, 'affineA') and not torch.all(torch.eq(self.affineA,torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda']))) ) ): # run this if do_affine == 1 or affineA exists and isn't identity
                if self.params['checkaffinestep'] == 1:
                    # new diffeo with old affine
                    # this doesn't match up with EAll even when vt is identity
                    phiinv0_temp,phiinv1_temp = self.forwardDeformationAffineVectorized2d(self.lastaffineA.clone(),phiinv0_gpu,phiinv1_gpu)
                    I = [None]*len(self.I)
                    for i in range(len(self.I)):
                        if self.params['v_scale'] != 1.0:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                        else:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                    
                    self.EMDiffeo.append( self.calculateMatchingEnergyMSEOnly2d(I) )
                    # new diffeo with new L and old T
                    phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineR2d(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu)
                    phiinv0_temp,phiinv1_temp = self.forwardDeformationAffineT2d(self.lastaffineA.clone(),phiinv0_gpu,phiinv1_gpu)
                    I = [None]*len(self.I)
                    for i in range(len(self.I)):
                        if self.params['v_scale'] != 1.0:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_temp.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                        else:
                            I[i] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                    
                    self.EMAffineR.append( self.calculateMatchingEnergyMSEOnly2d(I) )
                    # new everything
                    phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineT2d(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu)
                    del phiinv0_temp,phiinv1_temp,phiinv2_temp
                else:
                    phiinv0_gpu,phiinv1_gpu = self.forwardDeformationAffineVectorized2d(self.affineA.clone(),phiinv0_gpu,phiinv1_gpu)
            
            # deform the image
            for i in range(len(self.I)):
                if self.params['v_scale'] != 1.0:
                    self.It[i][t+1] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((torch.squeeze(torch.nn.functional.interpolate(phiinv1_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[1]*self.dx[1]-self.dx[1])*2,torch.squeeze(torch.nn.functional.interpolate(phiinv0_gpu.unsqueeze(0).unsqueeze(0),size=(self.nx[0],self.nx[1]),mode='bilinear',align_corners=True))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
                else:
                    self.It[i][t+1] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
        
        del phiinv0_gpu, phiinv1_gpu
    
    ## deform template forward
    #def forwardDeformation2d(self):
    #    phiinv0_gpu = self.X0.clone()
    #    phiinv1_gpu = self.X1.clone()
    #    # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
    #    for t in range(self.params['nt']):
    #        # update phiinv using method of characteristics
    #        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
    #        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)            
    #        '''
    #        if t == self.params['nt']-1 and self.params['do_affine'] == 1:
    #            if self.params['checkaffinestep'] == 1:
    #                # new diffeo with old affine
    #                phiinv0_temp,phiinv1_temp,phiinv2_temp = self.forwardDeformationAffineVectorized(self.lastaffineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
    #                I = torch.squeeze(grid_sample(self.It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_temp/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border'))
    #                self.EMDiffeo.append( self.calculateMatchingEnergyMSEOnly(I) )
    #                # new diffeo with new L and old T
    #                phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineR(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
    #                phiinv0_temp,phiinv1_temp,phiinv2_temp = self.forwardDeformationAffineT(self.lastaffineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
    #                I = torch.squeeze(grid_sample(self.It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_temp/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border'))
    #                self.EMAffineR.append( self.calculateMatchingEnergyMSEOnly(I) )
    #                # new everything
    #                phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineT(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
    #            else:
    #                phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineVectorized(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
    #        '''
    #        # deform the image
    #        for i in range(len(self.I)):
    #            self.It[i][t+1] = torch.squeeze(grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))
    #    
    #    return self.It,phiinv0_gpu, phiinv1_gpu
    
    # deform template forward using affine transform
    # this could be vectorized by stacking X0, X1, X2
    def forwardDeformationAffine(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        affineB = torch.inverse(affineA)
        
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3]
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3]
        #Zs = affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3]
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])
        phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])
        '''
        #Xs = affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3]
        #Ys = affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3]
        #Zs = affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3]
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0))) + (affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3])
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0))) + (affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3])
        phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,(affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,(affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0))) + (affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3])
        '''
        # deform the last time point in the image list
        # actually, just compute phiinv0_gpu, phiinv1_gpu, etc, before the final time step's image deformation
        #self.It[-1] = torch.squeeze(grid_sample(self.It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0)))
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # deform template forward using affine transform vectorized
    def forwardDeformationAffineVectorized(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu,interpmode='bilinear'):
        #affineA = affineA[[1,0,2,3],:]
        #affineA = affineA[:,[1,0,2,3]]
        affineB = torch.inverse(affineA)
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3]
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3]
        #Zs = affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3]
        s = torch.mm(affineB[0:3,0:3],torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,)),torch.reshape(self.X2,(-1,))), dim=0)) + torch.reshape(affineB[0:3,3],(3,1)).expand(-1,self.X0.numel())
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border',mode=interpmode)) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border',mode=interpmode)) + (torch.reshape(s[1,:],(self.X1.shape)))
        phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border',mode=interpmode)) + (torch.reshape(s[2,:],(self.X2.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # deform template forward using affine transform vectorized
    def forwardDeformationAffineVectorized2d(self,affineA,phiinv0_gpu,phiinv1_gpu,interpmode='bilinear'):
        #affineA = affineA[[1,0,2,3],:]
        #affineA = affineA[:,[1,0,2,3]]
        affineB = torch.inverse(affineA)
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]
        s = torch.mm(affineB[0:2,0:2],torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,))), dim=0)) + torch.reshape(affineB[0:2,2],(2,1)).expand(-1,self.X0.numel())
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border',mode=interpmode)) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border',mode=interpmode)) + (torch.reshape(s[1,:],(self.X1.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu
    
    # deform template forward using affine without translation
    def forwardDeformationAffineR(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        affineB = torch.inverse(affineA)
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2
        #Zs = affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2
        s = torch.mm(affineB[0:3,0:3],torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,)),torch.reshape(self.X2,(-1,))), dim=0))
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[2,:],(self.X2.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # deform template forward using affine without translation
    def forwardDeformationAffineR2d(self,affineA,phiinv0_gpu,phiinv1_gpu):
        affineB = torch.inverse(affineA)
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1
        s = torch.mm(affineB[0:2,0:2],torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,))), dim=0))
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu
    
    # deform template forward using affine translation
    def forwardDeformationAffineT2d(self,affineA,phiinv0_gpu,phiinv1_gpu):
        affineB = torch.inverse(affineA)
        s = torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,))), dim=0) + torch.reshape(affineB[0:2,2],(2,1)).expand(-1,self.X0.numel())
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu
    
    # deform template forward using affine translation
    def forwardDeformationAffineT(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        affineB = torch.inverse(affineA)
        s = torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,)),torch.reshape(self.X2,(-1,))), dim=0) + torch.reshape(affineB[0:3,3],(3,1)).expand(-1,self.X0.numel())
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[2,:],(self.X2.shape)))
        del s
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # compute contrast correction values
    # NOTE: does not subsample image for SGD for now
    def computeLinearContrastTransform(self,I,J,weight=1.0):
        Ibar = torch.sum(I*weight*self.M)/torch.sum(weight*self.M)
        Jbar = torch.sum(J*weight*self.M)/torch.sum(weight*self.M)
        VarI = torch.sum(((I-Ibar)*weight*self.M)**2)/torch.sum(weight*self.M)
        CovIJ = torch.sum((I-Ibar)*(J-Jbar)*weight*self.M)/torch.sum(weight*self.M)
        return Ibar, Jbar, VarI, CovIJ
    
    # contrast correction convenience function
    def runContrastCorrection(self):
        for i in self.params['cc_channels']:
            if i in self.params['we_channels'] and self.params['we'] != 0:
                if self.params['low_memory'] == 0:
                    self.ccIbar[i],self.ccJbar[i],self.ccVarI[i],self.ccCovIJ[i] = self.computeLinearContrastTransform(self.It[i][-1], self.J[i],self.W[i][0])
                else:
                    self.ccIbar[i],self.ccJbar[i],self.ccVarI[i],self.ccCovIJ[i] = self.computeLinearContrastTransform(self.applyThisTransformNT(self.I[i],nt=self.params['nt']), self.J[i],self.W[i][0])
            else:
                if self.params['low_memory'] == 0:
                    self.ccIbar[i],self.ccJbar[i],self.ccVarI[i],self.ccCovIJ[i] = self.computeLinearContrastTransform(self.It[i][-1], self.J[i])
                else:
                    self.ccIbar[i],self.ccJbar[i],self.ccVarI[i],self.ccCovIJ[i] = self.computeLinearContrastTransform(self.applyThisTransformNT(self.I[i],nt=self.params['nt']), self.J[i])
        
        return
    
    def applyContrastCorrection(self,I,i):
        #return [ ((x - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]) for i,x in enumerate(I)]
        return ((I - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i])
    
    # compute weight estimation
    def computeWeightEstimation(self):
        for ii in range(self.params['we']):
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    if ii == 0:
                        if self.params['low_memory'] == 0:
                            self.W[i][ii] = 1.0/np.sqrt(2.0*np.pi*self.params['sigma'][i]**2) * torch.exp(-1.0/2.0/self.params['sigma'][i]**2 * (self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2)
                        else:
                            self.W[i][ii] = 1.0/np.sqrt(2.0*np.pi*self.params['sigma'][i]**2) * torch.exp(-1.0/2.0/self.params['sigma'][i]**2 * (self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=self.params['nt']),i) - self.J[i])**2)
                    else:
                        self.W[i][ii] = 1.0/np.sqrt(2.0*np.pi*self.params['sigmaW'][ii]**2) * torch.exp(-1.0/2.0/self.params['sigmaW'][ii]**2 * (self.we_C[i][ii] - self.J[i])**2)
        
        for i in range(len(self.I)):
            if self.J[0].dim() == 3:
                Wsum = torch.sum(torch.stack(self.W[i],3),3)
            elif self.J[0].dim() == 2:
                Wsum = torch.sum(torch.stack(self.W[i],2),2)
            
            for ii in range(self.params['we']):
                self.W[i][ii] = self.W[i][ii] / Wsum
        
        del Wsum
        return
    
    # update weight estimation constants
    def updateWeightEstimationConstants(self):
        for i in range(len(self.I)):
            if i in self.params['we_channels']:
                for ii in range(self.params['we']):
                    self.we_C[i][ii] = torch.sum(self.W[i][ii] * self.J[i]) / torch.sum(self.W[i][ii])
        
        return
    
    # compute regularization energy for time varying velocity field in for loop to conserve memory
    def calculateRegularizationEnergyVt(self):
        ER = 0.0
        for t in range(self.params['nt']):
            # rfft produces a 2 channel matrix, torch does not support complex number multiplication yet
            ER += torch.sum(self.vt0[t]*torch.irfft(torch.rfft(self.vt0[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False) + self.vt1[t]*torch.irfft(torch.rfft(self.vt1[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False) + self.vt2[t]*torch.irfft(torch.rfft(self.vt2[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False)) * 0.5 / self.params['sigmaR']**2 * self.dx[0]*self.dx[1]*self.dx[2]*self.dt / self.params['v_scale']**3
        
        return ER
    
    # compute regularization energy for time varying velocity field in for loop to conserve memory
    def calculateRegularizationEnergyVt2d(self):
        ER = 0.0
        for t in range(self.params['nt']):
            # rfft produces a 2 channel matrix, torch does not support complex number multiplication yet
            ER += torch.sum(self.vt0[t]*torch.irfft(torch.rfft(self.vt0[t],2,onesided=False)*(1.0/self.Khat),2,onesided=False) + self.vt1[t]*torch.irfft(torch.rfft(self.vt1[t],2,onesided=False)*(1.0/self.Khat),2,onesided=False)) * 0.5 / self.params['sigmaR']**2 * self.dx[0]*self.dx[1]*self.dt
        
        return ER
    
    # compute matching energy
    def calculateMatchingEnergyMSE(self):
        lambda1 = [None]*len(self.I)
        EM = 0
        if self.params['we'] == 0:
            for i in range(len(self.I)):
                if self.params['low_memory'] == 0:
                    lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                    EM += torch.sum(self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                else:
                    lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                    EM += torch.sum(self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                
                if self.params['optimizer'] == 'sgd':
                    lambda1[i] *= self.sgd_M
        else:
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    for ii in range(self.params['we']):
                        if ii == 0:
                            if self.params['low_memory'] == 0:
                                lambda1[i] = -1*self.W[i][ii]*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2
                                EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                            else:
                                lambda1[i] = -1*self.W[i][ii]*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2
                                EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                            
                            if self.params['optimizer'] == 'sgd':
                                lambda1[i] *= self.sgd_M
                        else:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.we_C[i][ii] - self.J[i])**2/(2.0*self.params['sigmaW'][ii]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                else:
                    if self.params['low_memory'] == 0:
                        lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                        EM += torch.sum(self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                    else:
                        lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                        EM += torch.sum(self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                    
                    if self.params['optimizer'] == 'sgd':
                        lambda1[i] *= self.sgd_M
        
        return lambda1, EM
    
    # compute matching energy
    def calculateMatchingEnergyMSE2d(self):
        lambda1 = [None]*len(self.I)
        EM = 0
        if self.params['we'] == 0:
            for i in range(len(self.I)):
                if self.params['low_memory'] == 0:
                    lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                    EM += torch.sum(self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                else:
                    lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                    EM += torch.sum(self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                
                if self.params['optimizer'] == 'sgd':
                    lambda1[i] *= self.sgd_M
        else:
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    for ii in range(self.params['we']):
                        if ii == 0:
                            if self.params['low_memory'] == 0:
                                lambda1[i] = -1*self.W[i][ii]*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2
                                EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                            else:
                                lambda1[i] = -1*self.W[i][ii]*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2
                                EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                            
                            if self.params['optimizer'] == 'sgd':
                                lambda1[i] *= self.sgd_M
                        else:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.we_C[i][ii] - self.J[i])**2/(2.0*self.params['sigmaW'][ii]**2))*self.dx[0]*self.dx[1]
                else:
                    if self.params['low_memory'] == 0:
                        lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                        EM += torch.sum(self.M*( self.applyContrastCorrection(self.It[i][-1],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                    else:
                        lambda1[i] = -1*self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
                        EM += torch.sum(self.M*( self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                    
                    if self.params['optimizer'] == 'sgd':
                        lambda1[i] *= self.sgd_M
        
        return lambda1, EM
    
    '''
    # compute matching energy
    def calculateMatchingEnergyMSE2d(self):
        lambda1 = [None]*len(self.I)
        EM = 0
        for i in range(len(self.I)):
            lambda1[i] = -1*self.M*( ((self.It[i][-1] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]) - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
            EM += torch.sum(self.M*( ((self.It[i][-1] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
        return lambda1, EM
    '''
    
    # compute matching energy without lambda1
    def calculateMatchingEnergyMSEOnly(self, I):
        EM = 0
        if self.params['we'] == 0:
            for i in range(len(self.I)):
                EM += torch.sum(self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
        else:
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    for ii in range(self.params['we']):
                        if ii == 0:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                        else:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.we_C[i][ii] - self.J[i])**2/(2.0*self.params['sigmaW'][ii]**2))*self.dx[0]*self.dx[1]*self.dx[2]
                else:
                    EM += torch.sum(self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
            
        return EM
    
    # compute matching energy without lambda1
    def calculateMatchingEnergyMSEOnly2d(self, I):
        EM = 0
        if self.params['we'] == 0:
            for i in range(len(self.I)):
                EM += torch.sum(self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
        else:
            for i in range(len(self.I)):
                if i in self.params['we_channels']:
                    for ii in range(self.params['we']):
                        if ii == 0:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
                        else:
                            EM += torch.sum(self.W[i][ii]*self.M*( self.we_C[i][ii] - self.J[i])**2/(2.0*self.params['sigmaW'][ii]**2))*self.dx[0]*self.dx[1]
                else:
                    EM += torch.sum(self.M*( self.applyContrastCorrection(I[i],i) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
            
        return EM
    
    '''
    # compute matching energy without lambda1
    def calculateMatchingEnergyMSEOnly2d(self, I):
        EM = 0
        for i in range(len(self.I)):
            EM += torch.sum(self.M*( ((I[i] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]) - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
        return EM
    '''
    
    # update sgd subsampling mask
    def updateSGDMask(self):
        self.sgd_maskiter += 1
        if self.sgd_maskiter == self.params['sg_holdcount']:
            self.sgd_maskiter = 0
        else:
            return
        
        if self.params['sg_mask_mode'] == 'gauss':
            self.sgd_M = mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'] == '2gauss':
            self.sgd_M = mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])) + mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'] == '3gauss':
            self.sgd_M = mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])) + mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])) + mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'] == 'rand':
            self.sgd_M = self.params['sg_rand_scale']*torch.rand((self.nx[0],self.nx[1],self.nx[2])).type(self.params['dtype']).to(device=self.params['cuda'])
        elif self.params['sg_mask_mode'] == 'binrand':
            self.sgd_M = torch.round(self.params['sg_rand_scale']*torch.rand((self.nx[0],self.nx[1],self.nx[2])).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'][0:5] == 'gauss' and len(self.params['sg_mask_mode']) > 5:
            self.sgd_M = mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
            for i in range(int(self.params['sg_mask_mode'][5:])-1):
                self.sgd_M += mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']))
        elif self.params['sg_mask_mode'][0:8] == 'bingauss' and len(self.params['sg_mask_mode']) > 8:
            self.sgd_M = torch.round(mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])))
            for i in range(int(self.params['sg_mask_mode'][8:])-1):
                self.sgd_M += torch.round(mygaussian_3d_torch_selectcenter_meshgrid(self.X0,self.X1,self.X2,self.params['sg_sigma'],((torch.rand(1)-0.5)*self.nx[0]*self.dx[0]/2.0).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[1]*self.dx[1]/2).type(self.params['dtype']).to(device=self.params['cuda']),((torch.rand(1)-0.5)*self.nx[2]*self.dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda'])))
            
    
    # update learning rate for gradient descent
    def updateGDLearningRate(self):
        flag = False
        if len(self.EAll) > 1:
            if self.params['optimizer'] == 'gdr':
                if self.params['checkaffinestep'] == 0 and self.params['do_affine'] == 0:
                    # energy increased
                    if self.EAll[-1] >= self.EAll[-2] or self.EAll[-1]/self.EAll[-2] > 0.99999:
                        self.GDBeta *= 0.7
                elif self.params['checkaffinestep'] == 0 and self.params['do_affine'] > 0:
                    # energy increased
                    if self.EAll[-1] >= self.EAll[-2] or self.EAll[-1]/self.EAll[-2] > 0.99999:
                        if self.params['do_lddmm'] == 1:
                            self.GDBeta *= 0.7
                        
                        self.GDBetaAffineR *= 0.7
                        self.GDBetaAffineT *= 0.7
                elif self.params['checkaffinestep'] == 1 and self.params['do_affine'] > 0:
                    # if diffeo energy increased
                    if self.ERAll[-1] + self.EMDiffeo[-1] > self.EAll[-2]:
                        self.GDBeta *= 0.7
                    
                    if self.EMAffineR[-1] > self.EMDiffeo[-1]:
                        self.GDBetaAffineR *= 0.7
                    
                    if self.EMAffineT[-1] > self.EMAffineR[-1]:
                        self.GDBetaAffineT *= 0.7
            elif self.params['optimizer'] == 'sgd' and len(self.EAll) >= self.params['sg_climbcount']:
                climbcheck = 0
                if self.params['checkaffinestep'] == 0 and self.params['do_affine'] == 0:
                    # energy increased
                    while climbcheck < self.params['sg_climbcount']:
                        if self.EAll[-1-climbcheck] >= self.EAll[-2-climbcheck] or self.EAll[-1-climbcheck]/self.EAll[-2-climbcheck] > 0.99999:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                self.GDBeta *= 0.7
                                break
                        else:
                            break
                elif self.params['checkaffinestep'] == 0 and self.params['do_affine'] > 0:
                    # energy increased
                    while climbcheck < self.params['sg_climbcount']:
                        if self.EAll[-1-climbcheck] >= self.EAll[-2-climbcheck] or self.EAll[-1-climbcheck]/self.EAll[-2-climbcheck] > 0.99999:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                if self.params['do_lddmm'] == 1:
                                    self.GDBeta *= 0.7

                                self.GDBetaAffineR *= 0.7
                                self.GDBetaAffineT *= 0.7
                                break
                        else:
                            break
                elif self.params['checkaffinestep'] == 1 and self.params['do_affine'] > 0:
                    # if diffeo energy increased
                    while climbcheck < self.params['sg_climbcount']:
                        if self.ERAll[-1-climbcheck] + self.EMDiffeo[-1-climbcheck] > self.EAll[-2-climbcheck]:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                self.GDBeta *= 0.7
                                break
                            else:
                                break
                        
                        if self.EMAffineR[-1-climbcheck] > self.EMDiffeo[-1-climbcheck]:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                self.GDBetaAffineR *= 0.7
                                break
                            else:
                                break

                        if self.EMAffineT[-1-climbcheck] > self.EMAffineR[-1-climbcheck]:
                            climbcheck += 1
                            if climbcheck == self.params['sg_climbcount']:
                                self.GDBetaAffineT *= 0.7
                                break
                            else:
                                break
            
            elif self.params['optimizer'] == 'gdw':
                # energy increased
                if self.EAll[-1] > self.EAll[-2]:
                    self.climbcount += 1
                    if self.climbcount > self.params['maxclimbcount']:
                        flag = True
                        self.GDBeta *= 0.7
                        self.climbcount = 0
                        self.vt0 = [x.to(device=self.params['cuda']) for x in self.best['vt0']]
                        self.vt1 = [x.to(device=self.params['cuda']) for x in self.best['vt1']]
                        if self.J[0].dim() > 2:
                            self.vt2 = [x.to(device=self.params['cuda']) for x in self.best['vt2']]
                        print('Reducing epsilon to ' + str((self.GDBeta*self.params['epsilon']).item()) + ' and resetting to last best point.')
                # energy decreased
                elif self.EAll[-1] < self.bestE:
                    self.climbcount = 0
                    self.GDBeta *= 1.04
                elif self.EAll[-1] < self.EAll[-2]:
                    self.climbcount = 0
        
        if self.params['savebestv']:
            if self.EAll[-1] < self.bestE:
                self.bestE = self.EAll[-1]
                # TODO: this may be too slow to keep doing on cpu. possibly clone on gpu and eat memory
                self.best['vt0'] = [x.cpu() for x in self.vt0]
                self.best['vt1'] = [x.cpu() for x in self.vt1]
                if self.J[0].dim() > 2:
                    self.best['vt2'] = [x.cpu() for x in self.vt2]
        
        return flag
    
    
    # compute gradient of affine transformation
    def calculateGradientA(self,affineA,lambda1,mode='affine'):
        self.gradA = torch.tensor(np.zeros((4,4))).type(self.params['dtype']).to(device=self.params['cuda'])
        affineB = torch.inverse(affineA)
        gi_x = [None]*len(self.I)
        gi_y = [None]*len(self.I)
        gi_z = [None]*len(self.I)
        #if self.params['v_scale'] != 1:
        #    grad_divisor_x_scale = np.ones(self.X0.shape)
        #    grad_divisor_x_scale[1:-1,:,:] = 2
        #    grad_divisor_x_scale = torch.tensor(grad_divisor_x_scale).type(self.params['dtype']).to(device=self.params['cuda'])
        #    grad_divisor_y_scale = np.ones(self.X0.shape)
        #    grad_divisor_y_scale[:,1:-1,:] = 2
        #    grad_divisor_y_scale = torch.tensor(grad_divisor_y_scale).type(self.params['dtype']).to(device=self.params['cuda'])
        #    grad_divisor_z_scale = np.ones(self.X0.shape)
        #    grad_divisor_z_scale[:,:,1:-1] = 2
        #    grad_divisor_z_scale = torch.tensor(grad_divisor_z_scale).type(self.params['dtype']).to(device=self.params['cuda'])
        for i in range(len(self.I)):
            if self.params['low_memory'] == 0:
                if self.params['v_scale'] != 1.0:
                    gi_x[i],gi_y[i],gi_z[i] = self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][-1],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))
                else:
                    gi_x[i],gi_y[i],gi_z[i] = self.torch_gradient(self.applyContrastCorrection(self.It[i][-1],i),self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
            else:
                if self.params['v_scale'] != 1.0:
                    gi_x[i],gi_y[i],gi_z[i] = self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))
                else:
                    gi_x[i],gi_y[i],gi_z[i] = self.torch_gradient(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i),self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
            # TODO: can this be efficiently vectorized?
            for r in range(3):
                for c in range(4):
                    # allocating on the fly, not good
                    dA = torch.tensor(np.zeros((4,4))).type(self.params['dtype']).to(device=self.params['cuda'])
                    dA[r,c] = 1.0
                    AdAB = torch.mm(torch.mm(affineA,dA),affineB)
                    #AdABX = AdAB[0,0]*self.X0 + AdAB[0,1]*self.X1 + AdAB[0,2]*self.X2 + AdAB[0,3]
                    #AdABY = AdAB[1,0]*self.X0 + AdAB[1,1]*self.X1 + AdAB[1,2]*self.X2 + AdAB[1,3]
                    #AdABZ = AdAB[2,0]*self.X0 + AdAB[2,1]*self.X1 + AdAB[2,2]*self.X2 + AdAB[2,3]
                    if i == 0:
                        if self.params['v_scale'] != 1.0:
                            self.gradA[r,c] = torch.sum( torch.squeeze(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)) * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]*(self.X2) + AdAB[0,3]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]*(self.X2) + AdAB[1,3]) + gi_z[i]*(AdAB[2,0]*(self.X0) + AdAB[2,1]*(self.X1) + AdAB[2,2]*(self.X2) + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
                        else:
                            self.gradA[r,c] = torch.sum( lambda1[i] * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]*(self.X2) + AdAB[0,3]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]*(self.X2) + AdAB[1,3]) + gi_z[i]*(AdAB[2,0]*(self.X0) + AdAB[2,1]*(self.X1) + AdAB[2,2]*(self.X2) + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
                    else:
                        if self.params['v_scale'] != 1.0:
                            self.gradA[r,c] += torch.sum( torch.squeeze(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)) * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]*(self.X2) + AdAB[0,3]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]*(self.X2) + AdAB[1,3]) + gi_z[i]*(AdAB[2,0]*(self.X0) + AdAB[2,1]*(self.X1) + AdAB[2,2]*(self.X2) + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
                        else:
                            self.gradA[r,c] += torch.sum( lambda1[i] * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]*(self.X2) + AdAB[0,3]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]*(self.X2) + AdAB[1,3]) + gi_z[i]*(AdAB[2,0]*(self.X0) + AdAB[2,1]*(self.X1) + AdAB[2,2]*(self.X2) + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
                    #self.gradA[r,c] = torch.sum( lambda1 * ( gi_y*(AdAB[0,0]*self.X1 + AdAB[0,1]*self.X0 + AdAB[0,2]*self.X2 + AdAB[0,3]) + gi_x*(AdAB[1,0]*self.X1 + AdAB[1,1]*self.X0 + AdAB[1,2]*self.X2 + AdAB[1,3]) + gi_z*(AdAB[2,0]*self.X1 + AdAB[2,1]*self.X0 + AdAB[2,2]*self.X2 + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
        
        # if rigid
        if mode == 'rigid':
            self.gradA -= torch.transpose(self.gradA,0,1)
        
        #if self.params['v_scale'] != 1:
        #    del grad_divisor_x_scale
        #    del grad_divisor_y_scale
        #    del grad_divisor_z_scale
        #    if self.params['low_memory'] == 1:
        #        torch.cuda.empty_cache()
    
    
    # compute gradient of affine transformation
    def calculateGradientA2d(self,affineA,lambda1,mode='affine'):
        self.gradA = torch.tensor(np.zeros((3,3))).type(self.params['dtype']).to(device=self.params['cuda'])
        affineB = torch.inverse(affineA)
        gi_x = [None]*len(self.I)
        gi_y = [None]*len(self.I)
        for i in range(len(self.I)):
            if self.params['low_memory'] == 0:
                if self.params['v_scale'] != 1.0:
                    gi_x[i],gi_y[i] = self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][-1],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))
                else:
                    gi_x[i],gi_y[i] = self.torch_gradient2d(self.applyContrastCorrection(self.It[i][-1],i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
            else:
                if self.params['v_scale'] != 1.0:
                    gi_x[i],gi_y[i] = self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))
                else:
                    gi_x[i],gi_y[i] = self.torch_gradient2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i]),i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
            # TODO: can this be efficiently vectorized?
            for r in range(2):
                for c in range(3):
                    # allocating on the fly, not good
                    dA = torch.tensor(np.zeros((3,3))).type(self.params['dtype']).to(device=self.params['cuda'])
                    dA[r,c] = 1.0
                    AdAB = torch.mm(torch.mm(affineA,dA),affineB)
                    #AdABX = AdAB[0,0]*self.X0 + AdAB[0,1]*self.X1 + AdAB[0,2]
                    #AdABY = AdAB[1,0]*self.X0 + AdAB[1,1]*self.X1 + AdAB[1,2]
                    if i == 0:
                        if self.params['v_scale'] != 1.0:
                            self.gradA[r,c] = torch.sum( torch.squeeze(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)) * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]) ) ) * self.dx[0]*self.dx[1]
                        else:
                            self.gradA[r,c] = torch.sum( lambda1[i] * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]) ) ) * self.dx[0]*self.dx[1]
                    else:
                        if self.params['v_scale'] != 1.0:
                            self.gradA[r,c] += torch.sum( torch.squeeze(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)) * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]) ) ) * self.dx[0]*self.dx[1]
                        else:
                            self.gradA[r,c] += torch.sum( lambda1[i] * ( gi_x[i]*(AdAB[0,0]*(self.X0) + AdAB[0,1]*(self.X1) + AdAB[0,2]) + gi_y[i]*(AdAB[1,0]*(self.X0) + AdAB[1,1]*(self.X1) + AdAB[1,2]) ) ) * self.dx[0]*self.dx[1]
                    #self.gradA[r,c] = torch.sum( lambda1 * ( gi_y*(AdAB[0,0]*self.X1 + AdAB[0,1]*self.X0 + AdAB[0,2]*self.X2 + AdAB[0,3]) + gi_x*(AdAB[1,0]*self.X1 + AdAB[1,1]*self.X0 + AdAB[1,2]*self.X2 + AdAB[1,3]) + gi_z*(AdAB[2,0]*self.X1 + AdAB[2,1]*self.X0 + AdAB[2,2]*self.X2 + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
        
        # if rigid
        if mode == 'rigid':
            self.gradA -= torch.transpose(self.gradA,0,1)
    
    
    # compute gradient per time step for time varying velocity field parameterization
    def calculateGradientVt(self,lambda1,t,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        # update phiinv using method of characteristics, note "+" because we are integrating backward
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X0+self.vt0[t]*self.dt)
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X1+self.vt1[t]*self.dt)
        phiinv2_gpu = torch.squeeze(grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border')) + (self.X2+self.vt2[t]*self.dt)
        
        # find the determinant of Jacobian
        if self.params['v_scale'] != 1:
            phiinv0_0,phiinv0_1,phiinv0_2 = self.torch_gradient(phiinv0_gpu,self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))
            phiinv1_0,phiinv1_1,phiinv1_2 = self.torch_gradient(phiinv1_gpu,self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))
            phiinv2_0,phiinv2_1,phiinv2_2 = self.torch_gradient(phiinv2_gpu,self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))
        else:
            phiinv0_0,phiinv0_1,phiinv0_2 = self.torch_gradient(phiinv0_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
            phiinv1_0,phiinv1_1,phiinv1_2 = self.torch_gradient(phiinv1_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
            phiinv2_0,phiinv2_1,phiinv2_2 = self.torch_gradient(phiinv2_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
        
        detjac = phiinv0_0*(phiinv1_1*phiinv2_2 - phiinv1_2*phiinv2_1)\
            - phiinv0_1*(phiinv1_0*phiinv2_2 - phiinv1_2*phiinv2_0)\
            + phiinv0_2*(phiinv1_0*phiinv2_1 - phiinv1_1*phiinv2_0)
        
        del phiinv0_0,phiinv0_1,phiinv0_2,phiinv1_0,phiinv1_1,phiinv1_2,phiinv2_0,phiinv2_1,phiinv2_2
        # deform phiinv back by affine transform if asked for
        # is this accumulating?
        #if self.params['do_affine'] == 1:
        #    phiinv0_gpu = self.affineA[0,0]*phiinv0_gpu + self.affineA[0,1]*phiinv1_gpu + self.affineA[0,2]*phiinv2_gpu + self.affineA[0,3]
        #    phiinv1_gpu = self.affineA[1,0]*phiinv0_gpu + self.affineA[1,1]*phiinv1_gpu + self.affineA[1,2]*phiinv2_gpu + self.affineA[1,3]
        #    phiinv2_gpu = self.affineA[2,0]*phiinv0_gpu + self.affineA[2,1]*phiinv1_gpu + self.affineA[2,2]*phiinv2_gpu + self.affineA[2,3]
        
        for i in range(len(self.I)):
            # find lambda_t
            #if not hasattr(self, 'affineA') or torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))):
            if not hasattr(self, 'affineA'):
            #if self.params['do_affine'] == 0:
                if self.params['v_scale'] < 1.0:
                    if self.params['v_scale_smoothing'] == 1:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(torch.nn.functional.conv3d(lambda1[i].unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True), torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))*detjac
                    else:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True), torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))*detjac
                else:
                    lambdat = torch.squeeze(grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))*detjac
            else:
                if self.params['v_scale'] < 1.0:
                    if self.params['v_scale_smoothing'] == 1:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(torch.nn.functional.conv3d(lambda1[i].unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True), torch.stack((((self.affineA[2,0]*(phiinv0_gpu)) + (self.affineA[2,1]*(phiinv1_gpu)) + (self.affineA[2,2]*(phiinv2_gpu)) + self.affineA[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + (self.affineA[1,2]*(phiinv2_gpu)) + self.affineA[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + (self.affineA[0,2]*(phiinv2_gpu)) + self.affineA[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
                    else:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True), torch.stack((((self.affineA[2,0]*(phiinv0_gpu)) + (self.affineA[2,1]*(phiinv1_gpu)) + (self.affineA[2,2]*(phiinv2_gpu)) + self.affineA[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + (self.affineA[1,2]*(phiinv2_gpu)) + self.affineA[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + (self.affineA[0,2]*(phiinv2_gpu)) + self.affineA[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
                else:
                    lambdat = torch.squeeze(grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((((self.affineA[2,0]*(phiinv0_gpu)) + (self.affineA[2,1]*(phiinv1_gpu)) + (self.affineA[2,2]*(phiinv2_gpu)) + self.affineA[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2,((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + (self.affineA[1,2]*(phiinv2_gpu)) + self.affineA[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + (self.affineA[0,2]*(phiinv2_gpu)) + self.affineA[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
            
            # get the gradient of the image at this time
            # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
            if i == 0:
                if self.params['low_memory'] == 0:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            #TODO: alternatively, compute image gradient and then downsample after
                            grad_list = [x*lambdat for x in self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv3d(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))]
                        else:
                            grad_list = [x*lambdat for x in self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))]
                    else:
                        grad_list = [x*lambdat for x in self.torch_gradient(self.applyContrastCorrection(self.It[i][t],i),self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)]
                else:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [x*lambdat for x in self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv3d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))]
                        else:
                            grad_list = [x*lambdat for x in self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))]
                    else:
                        grad_list = [x*lambdat for x in self.torch_gradient(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i),self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)]
            else:
                if self.params['low_memory'] == 0:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv3d(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))])]
                        else:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))])]
                    else:
                        grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient(self.applyContrastCorrection(self.It[i][t],i),self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)])]
                else:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv3d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))])]
                        else:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),self.dx[0],self.dx[1],self.dx[2],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_z.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1],self.X0.shape[2]),mode='trilinear',align_corners=True)))])]
                    else:
                        grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i),self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)])]
        
        # smooth it
        del lambdat, detjac
        if self.params['low_memory'] > 0:
            torch.cuda.empty_cache()
            
        if self.params['optimizer'] != 'adam':
            grad_list = [torch.irfft(torch.rfft(x,3,onesided=False)*self.Khat,3,onesided=False) for x in grad_list]
            # add the regularization term
            grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
            grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
            grad_list[2] += self.vt2[t]/self.params['sigmaR']**2
        
        return grad_list,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu
    
    # compute gradient per time step for time varying velocity field parameterization
    def calculateGradientVt2d(self,lambda1,t,phiinv0_gpu,phiinv1_gpu):
        # update phiinv using method of characteristics, note "+" because we are integrating backward
        phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0+self.vt0[t]*self.dt)
        phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1+self.vt1[t]*self.dt)
        
        # find the determinant of Jacobian
        if self.params['v_scale'] != 1:
            phiinv0_0,phiinv0_1 = self.torch_gradient2d(phiinv0_gpu,self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))
            phiinv1_0,phiinv1_1 = self.torch_gradient2d(phiinv1_gpu,self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))
        else:
            phiinv0_0,phiinv0_1 = self.torch_gradient2d(phiinv0_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
            phiinv1_0,phiinv1_1 = self.torch_gradient2d(phiinv1_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
        
        detjac = phiinv0_0*phiinv1_1 - phiinv0_1*phiinv1_0
        self.detjac[t] = detjac.clone()
        
        del phiinv0_0,phiinv0_1,phiinv1_0,phiinv1_1
        # deform phiinv back by affine transform if asked for
        # is this accumulating?
        #if self.params['do_affine'] == 1:
        #    phiinv0_gpu = self.affineA[0,0]*phiinv0_gpu + self.affineA[0,1]*phiinv1_gpu + self.affineA[0,2]*phiinv2_gpu + self.affineA[0,3]
        #    phiinv1_gpu = self.affineA[1,0]*phiinv0_gpu + self.affineA[1,1]*phiinv1_gpu + self.affineA[1,2]*phiinv2_gpu + self.affineA[1,3]
        #    phiinv2_gpu = self.affineA[2,0]*phiinv0_gpu + self.affineA[2,1]*phiinv1_gpu + self.affineA[2,2]*phiinv2_gpu + self.affineA[2,3]
        
        for i in range(len(self.I)):
            # find lambda_t
            #if not hasattr(self, 'affineA') or torch.all(torch.eq(self.affineA,torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda']))):
            if not hasattr(self, 'affineA'):
            #if self.params['do_affine'] == 0:
                if self.params['v_scale'] < 1.0:
                    if self.params['v_scale_smoothing'] == 1:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(torch.nn.functional.conv2d(lambda1[i].unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac
                    else:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac
                else:
                    lambdat = torch.squeeze(grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac
            else:
                if self.params['v_scale'] < 1.0:
                    if self.params['v_scale_smoothing'] == 1:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(torch.nn.functional.conv2d(lambda1[i].unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True), torch.stack((((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + self.affineA[1,2])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + self.affineA[0,2])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
                    else:
                        lambdat = torch.squeeze(grid_sample(torch.nn.functional.interpolate(lambda1[i].unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True), torch.stack((((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + self.affineA[1,2])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + self.affineA[0,2])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
                else:
                    lambdat = torch.squeeze(grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((((self.affineA[1,0]*(phiinv0_gpu)) + (self.affineA[1,1]*(phiinv1_gpu)) + self.affineA[1,2])/(self.nx[1]*self.dx[1]-self.dx[1])*2,((self.affineA[0,0]*(phiinv0_gpu)) + (self.affineA[0,1]*(phiinv1_gpu)) + self.affineA[0,2])/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac*torch.abs(torch.det(self.affineA))
            
            # get the gradient of the image at this time
            # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
            if i == 0:
                if self.params['low_memory'] == 0:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            #TODO: alternatively, compute image gradient and then downsample after
                            grad_list = [x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv2d(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))]
                        else:
                            grad_list = [x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))]
                    else:
                        grad_list = [x*lambdat for x in self.torch_gradient2d(self.applyContrastCorrection(self.It[i][t],i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)]
                else:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))]
                        else:
                            grad_list = [x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))]
                    else:
                        grad_list = [x*lambdat for x in self.torch_gradient2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)]
            else:
                if self.params['low_memory'] == 0:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv2d(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))])]
                        else:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.It[i][t],i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))])]
                    else:
                        grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(self.applyContrastCorrection(self.It[i][t],i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)])]
                else:
                    if self.params['v_scale'] != 1.0:
                        if self.params['v_scale_smoothing'] == 1:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(torch.nn.functional.conv2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),self.gaussian_filter.unsqueeze(0).unsqueeze(0), stride=1, padding = int(self.gaussian_filter.shape[0]/2.0)),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))])]
                        else:
                            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(torch.squeeze(torch.nn.functional.interpolate(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i).unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),self.dx[0],self.dx[1],torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_x.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)),torch.squeeze(torch.nn.functional.interpolate(self.grad_divisor_y.unsqueeze(0).unsqueeze(0),size=(self.X0.shape[0],self.X0.shape[1]),mode='bilinear',align_corners=True)))])]
                    else:
                        grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(self.applyContrastCorrection(self.applyThisTransformNT(self.I[i],nt=t),i),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)])]
        
        # smooth it
        del lambdat, detjac
        if self.params['low_memory'] > 0:
            torch.cuda.empty_cache()
            
        if self.params['optimizer'] != 'adam':
            grad_list = [torch.irfft(torch.rfft(x,2,onesided=False)*self.Khat,2,onesided=False) for x in grad_list]
            # add the regularization term
            grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
            grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
        
        return grad_list,phiinv0_gpu,phiinv1_gpu
    
    ## compute gradient per time step for time varying velocity field parameterization
    #def calculateGradientVt2d(self,lambda1,t,phiinv0_gpu,phiinv1_gpu):
    #    # update phiinv using method of characteristics, note "+" because we are integrating backward
    #    phiinv0_gpu = torch.squeeze(grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0+self.vt0[t]*self.dt)
    #    phiinv1_gpu = torch.squeeze(grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1+self.vt1[t]*self.dt)        
    #    
    #    # find the determinant of Jacobian
    #    phiinv0_0,phiinv0_1 = self.torch_gradient2d(phiinv0_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
    #    phiinv1_0,phiinv1_1 = self.torch_gradient2d(phiinv1_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
    #    detjac = phiinv0_0 * phiinv1_1 - phiinv0_1 * phiinv1_0
    #    self.detjac[t] = detjac.clone()
    #    '''
    #    # deform phiinv back by affine transform if asked for
    #    if self.params['do_affine'] == 1:
    #        phiinv0_gpu = self.affineA[0,0]*phiinv0_gpu + self.affineA[0,1]*phiinv1_gpu + self.affineA[0,2]
    #        phiinv1_gpu = self.affineA[1,0]*phiinv0_gpu + self.affineA[1,1]*phiinv1_gpu + self.affineA[1,2]
    #    '''
    #    for i in range(len(self.I)):
    #        # find lambda_t
    #        #if self.params['do_affine'] == 0:
    #        lambdat = torch.squeeze(grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=2).unsqueeze(0),padding_mode='zeros'))*detjac
    #        #else:
    #        #    lambdat = torch.squeeze(grid_sample(lambda1.unsqueeze(0).unsqueeze(0), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2),dim=3).unsqueeze(0),padding_mode='border'))*detjac*torch.det(self.affineA)
    #        # get the gradient of the image at this time
    #        # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
    #        if i == 0:
    #            grad_list = [x*lambdat for x in self.torch_gradient2d(((self.It[i][t] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)]
    #        else:
    #            grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(((self.It[i][t] - self.ccIbar[i])*self.ccCovIJ[i]/self.ccVarI[i] + self.ccJbar[i]),self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)])]
    #    
    #    # smooth it
    #    grad_list = [torch.irfft(torch.rfft(x,2,onesided=False)*self.Khat,2,onesided=False) for x in grad_list]
    #    
    #    # add the regularization term
    #    grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
    #    grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
    #    return grad_list,phiinv0_gpu,phiinv1_gpu
    
    # update gradient
    def updateGradientVt(self,t,grad_list,iter=0):
        if self.params['optimizer'] == 'adam':
            #self.vt0[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m0'][t] / (torch.sqrt(self.adam['v0'][t]) + self.params['adam_epsilon'])
            #self.vt1[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m1'][t] / (torch.sqrt(self.adam['v1'][t]) + self.params['adam_epsilon'])
            #if self.J[0].dim() > 2:
            #    self.vt2[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m2'][t] / (torch.sqrt(self.adam['v2'][t]) + self.params['adam_epsilon'])
            #self.vt0[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m0'][t] / ((self.adam['v0'][t]) + self.params['adam_epsilon'])
            #self.vt1[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m1'][t] / ((self.adam['v1'][t]) + self.params['adam_epsilon'])
            #if self.J[0].dim() > 2:
            #    self.vt2[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * self.adam['m2'][t] / ((self.adam['v2'][t]) + self.params['adam_epsilon'])
            self.vt0[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * (torch.irfft(torch.rfft(self.adam['m0'][t] / (torch.sqrt(self.adam['v0'][t]) + self.params['adam_epsilon']),2,onesided=False)*self.Khat,2,onesided=False)) + self.vt0[t]/self.params['sigmaR']**2
            self.vt1[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * (torch.irfft(torch.rfft(self.adam['m1'][t] / (torch.sqrt(self.adam['v1'][t]) + self.params['adam_epsilon']),2,onesided=False)*self.Khat,2,onesided=False)) + self.vt1[t]/self.params['sigmaR']**2
            if self.J[0].dim() > 2:
                self.vt2[t] -= self.params['adam_alpha']*(1-self.params['adam_beta2']**(iter+1))**(1/2) / (1-self.params['adam_beta1']**(iter+1)) * (torch.irfft(torch.rfft(self.adam['m2'][t] / (torch.sqrt(self.adam['v2'][t]) + self.params['adam_epsilon']),2,onesided=False)*self.Khat,2,onesided=False)) + self.vt2[t]/self.params['sigmaR']**2
        elif self.params['optimizer'] == "adadelta":
            self.vt0[t] -= torch.irfft(torch.rfft(torch.sqrt(self.adadelta['v0'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0],3,onesided=False)*self.Khat,3,onesided=False)
            self.vt1[t] -= torch.irfft(torch.rfft(torch.sqrt(self.adadelta['v1'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m1'][t] + self.params['ada_epsilon']) * grad_list[1],3,onesided=False)*self.Khat,3,onesided=False)
            if self.J[0].dim() > 2:
                self.vt2[t] -= torch.irfft(torch.rfft(torch.sqrt(self.adadelta['v2'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m2'][t] + self.params['ada_epsilon']) * grad_list[2],3,onesided=False)*self.Khat,3,onesided=False)
            self.adadelta['v0'][t] = self.params['ada_rho']*self.adadelta['v0'][t] + (1-self.params['ada_rho'])*(-1*torch.irfft(torch.rfft(torch.sqrt(self.adadelta['v0'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0],3,onesided=False)*self.Khat,3,onesided=False))**2
            self.adadelta['v1'][t] = self.params['ada_rho']*self.adadelta['v1'][t] + (1-self.params['ada_rho'])*(-1*torch.irfft(torch.rfft(torch.sqrt(self.adadelta['v1'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m1'][t] + self.params['ada_epsilon']) * grad_list[1],3,onesided=False)*self.Khat,3,onesided=False))**2
            if self.J[0].dim() > 2:
                self.adadelta['v2'][t] = self.params['ada_rho']*self.adadelta['v2'][t] + (1-self.params['ada_rho'])*(-1*torch.irfft(torch.rfft(torch.sqrt(self.adadelta['v2'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m2'][t] + self.params['ada_epsilon']) * grad_list[2],3,onesided=False)*self.Khat,3,onesided=False))**2
            #self.vt0[t] -= (self.adadelta['v0'][t] + self.params['ada_epsilon']) / (self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0]
            #self.vt1[t] -= (self.adadelta['v1'][t] + self.params['ada_epsilon']) / (self.adadelta['m1'][t] + self.params['ada_epsilon']) * grad_list[1]
            #self.vt2[t] -= (self.adadelta['v2'][t] + self.params['ada_epsilon']) / (self.adadelta['m2'][t] + self.params['ada_epsilon']) * grad_list[2]
            #self.adadelta['v0'][t] = self.params['ada_rho']*self.adadelta['v0'][t] + (1-self.params['ada_rho'])*(-1*(self.adadelta['v0'][t] + self.params['ada_epsilon']) / (self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0])**2
            #self.adadelta['v1'][t] = self.params['ada_rho']*self.adadelta['v1'][t] + (1-self.params['ada_rho'])*(-1*(self.adadelta['v1'][t] + self.params['ada_epsilon']) / (self.adadelta['m1'][t] + self.params['ada_epsilon']) * grad_list[1])**2
            #self.adadelta['v2'][t] = self.params['ada_rho']*self.adadelta['v2'][t] + (1-self.params['ada_rho'])*(-1*(self.adadelta['v2'][t] + self.params['ada_epsilon']) / (self.adadelta['m2'][t] + self.params['ada_epsilon']) * grad_list[2])**2
        elif self.params['optimizer'] == 'rmsprop':
            self.vt0[t] -= torch.irfft(torch.rfft(self.params['rms_alpha'] / torch.sqrt(self.rmsprop['m0'][t] + self.params['rms_epsilon']) * grad_list[0],3,onesided=False)*self.Khat,3,onesided=False) + self.vt0[t]/self.params['sigmaR']**2
            self.vt1[t] -= torch.irfft(torch.rfft(self.params['rms_alpha'] / torch.sqrt(self.rmsprop['m1'][t] + self.params['rms_epsilon']) * grad_list[1],3,onesided=False)*self.Khat,3,onesided=False) + self.vt1[t]/self.params['sigmaR']**2
            if self.J[0].dim() > 2:
                self.vt2[t] -= torch.irfft(torch.rfft(self.params['rms_alpha'] / torch.sqrt(self.rmsprop['m2'][t] + self.params['rms_epsilon']) * grad_list[2],3,onesided=False)*self.Khat,3,onesided=False) + self.vt2[t]/self.params['sigmaR']**2
        elif self.params['optimizer'] == 'sgdm':
            self.vt0[t] -= self.sgdm['m0'][t]
            self.vt1[t] -= self.sgdm['m1'][t]
            if self.J[0].dim() > 2:
                self.vt2[t] -= self.sgdm['m2'][t]
        else:
            self.vt0[t] -= self.params['epsilon']*self.GDBeta*grad_list[0]
            self.vt1[t] -= self.params['epsilon']*self.GDBeta*grad_list[1]
            if self.J[0].dim() > 2:
                self.vt2[t] -= self.params['epsilon']*self.GDBeta*grad_list[2]
    
    # update adam parameters
    def updateAdamLearningRate(self,t,grad_list):
        # don't normalize here, save normalization for the update step
        #self.adam['m0'][t] = [self.params['adam_beta1']*x + (1-self.params['adam_beta1'])*y for (x,y) in zip(self.adam['m0'],grad_list[0])]
        #self.adam['m1'][t] = [self.params['adam_beta1']*x + (1-self.params['adam_beta1'])*y for (x,y) in zip(self.adam['m1'],grad_list[1])]
        #self.adam['m2'][t] = [self.params['adam_beta1']*x + (1-self.params['adam_beta1'])*y for (x,y) in zip(self.adam['m2'],grad_list[2])]
        #self.adam['v0'][t] = [self.params['adam_beta2']*x + (1-self.params['adam_beta2'])*(y**2) for (x,y) in zip(self.adam['v0'],grad_list[0])]
        #self.adam['v1'][t] = [self.params['adam_beta2']*x + (1-self.params['adam_beta2'])*(y**2) for (x,y) in zip(self.adam['v1'],grad_list[1])]
        #self.adam['v2'][t] = [self.params['adam_beta2']*x + (1-self.params['adam_beta2'])*(y**2) for (x,y) in zip(self.adam['v2'],grad_list[2])]
        self.adam['m0'][t] = self.params['adam_beta1']*self.adam['m0'][t] + (1-self.params['adam_beta1'])*grad_list[0]
        self.adam['m1'][t] = self.params['adam_beta1']*self.adam['m1'][t] + (1-self.params['adam_beta1'])*grad_list[1]
        self.adam['m2'][t] = self.params['adam_beta1']*self.adam['m2'][t] + (1-self.params['adam_beta1'])*grad_list[2]
        self.adam['v0'][t] = self.params['adam_beta2']*self.adam['v0'][t] + (1-self.params['adam_beta2'])*(grad_list[0]**2)
        self.adam['v1'][t] = self.params['adam_beta2']*self.adam['v1'][t] + (1-self.params['adam_beta2'])*(grad_list[1]**2)
        self.adam['v2'][t] = self.params['adam_beta2']*self.adam['v2'][t] + (1-self.params['adam_beta2'])*(grad_list[2]**2)
    
    # update adadelta parameters
    def updateAdadeltaLearningRate(self,t,grad_list):
        # accumulate gradient
        self.adadelta['m0'][t] = self.params['ada_rho']*self.adadelta['m0'][t] + (1-self.params['ada_rho'])*grad_list[0]**2
        self.adadelta['m1'][t] = self.params['ada_rho']*self.adadelta['m1'][t] + (1-self.params['ada_rho'])*grad_list[1]**2
        self.adadelta['m2'][t] = self.params['ada_rho']*self.adadelta['m2'][t] + (1-self.params['ada_rho'])*grad_list[2]**2
        # accumulate parameter update
        #self.adadelta['v0'][t] = self.params['ada_rho']*self.adadelta['v0'][t] + (1-self.params['ada_rho'])*self.vt0[t]**2
        #self.adadelta['v1'][t] = self.params['ada_rho']*self.adadelta['v1'][t] + (1-self.params['ada_rho'])*self.vt1[t]**2
        #self.adadelta['v2'][t] = self.params['ada_rho']*self.adadelta['v2'][t] + (1-self.params['ada_rho'])*self.vt2[t]**2
        # this is the update step
        #-1 * torch.sqrt(self.adadelta['v0'][t] + self.params['ada_epsilon']) / torch.sqrt(self.adadelta['m0'][t] + self.params['ada_epsilon']) * grad_list[0]
    
    # update rmsprop parameters
    def updateRMSPropLearningRate(self,t,grad_list):
        self.rmsprop['m0'][t] = self.params['rms_rho']*self.rmsprop['m0'][t] + (1-self.params['rms_rho'])*grad_list[0]**2
        self.rmsprop['m1'][t] = self.params['rms_rho']*self.rmsprop['m1'][t] + (1-self.params['rms_rho'])*grad_list[1]**2
        self.rmsprop['m2'][t] = self.params['rms_rho']*self.rmsprop['m2'][t] + (1-self.params['rms_rho'])*grad_list[2]**2
    
    # update sgdm parameters
    def updateSGDMLearningRate(self,t,grad_list):
        # do I need GDBeta here?
        self.sgdm['m0'][t] = self.params['sg_gamma']*self.sgdm['m0'][t] + self.params['epsilon']*self.GDBeta*grad_list[0]
        self.sgdm['m1'][t] = self.params['sg_gamma']*self.sgdm['m1'][t] + self.params['epsilon']*self.GDBeta*grad_list[1]
        self.sgdm['m2'][t] = self.params['sg_gamma']*self.sgdm['m2'][t] + self.params['epsilon']*self.GDBeta*grad_list[2]
    
    # convenience function for calculating and updating gradients of Vt
    def calculateAndUpdateGradientsVt(self, lambda1, iter=0):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        if self.J[0].dim() > 2:
            phiinv2_gpu = self.X2.clone()
        
        for t in range(self.params['nt']-1,-1,-1):
            if self.J[0].dim() > 2:
                grad_list,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.calculateGradientVt(lambda1,t,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
            else:
                grad_list,phiinv0_gpu,phiinv1_gpu = self.calculateGradientVt2d(lambda1,t,phiinv0_gpu,phiinv1_gpu)
            
            if self.params['optimizer'] == 'adam':
                self.updateAdamLearningRate(t,grad_list)
            elif self.params['optimizer'] == 'adadelta':
                self.updateAdadeltaLearningRate(t,grad_list)
            elif self.params['optimizer'] == 'rmsprop':
                self.updateRMSPropLearningRate(t,grad_list)
            elif self.params['optimizer'] == 'sgdm':
                self.updateSGDMLearningRate(t,grad_list)
            
            self.updateGradientVt(t,grad_list,iter=iter)
            del grad_list
        
        del phiinv0_gpu,phiinv1_gpu
        if self.J[0].dim() > 2:
            del phiinv2_gpu
    
    # update affine matrix
    def updateAffine(self):
        # transfer to cpu for matrix exponential, takes about 20ms round trip
        gradA_cpu_numpy = self.gradA.cpu().numpy()
        e = np.zeros((4,4))
        e[0:3,0:3] = self.params['epsilonL']*self.GDBetaAffineR
        e[0:3,3] = self.params['epsilonT']*self.GDBetaAffineT
        e = torch.tensor(scipy.linalg.expm(-e * gradA_cpu_numpy)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.lastaffineA = self.affineA.clone()
        self.affineA = torch.mm(self.affineA,e)
    
    # update affine matrix
    def updateAffine2d(self):
        # transfer to cpu for matrix exponential, takes about 20ms round trip
        gradA_cpu_numpy = self.gradA.cpu().numpy()
        e = np.zeros((3,3))
        e[0:2,0:2] = self.params['epsilonL']*self.GDBetaAffineR
        e[0:2,2] = self.params['epsilonT']*self.GDBetaAffineT
        e = torch.tensor(scipy.linalg.expm(-e * gradA_cpu_numpy)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.lastaffineA = self.affineA.clone()
        self.affineA = torch.mm(self.affineA,e)
    
    # update epsilon after a run
    def updateEpsilonAfterRun(self):
        self.setParams('epsilonL',self.GDBetaAffineR*self.params['epsilonL']) # reduce step size, here we set it to the current size
        self.setParams('epsilonT',self.GDBetaAffineT*self.params['epsilonT']) # reduce step size, here we set it to the current size
    
    # for section alignment
    def sa(self, input, atlas, a=None, b=None, theta=None, dx=None, nx=None, niter=250, dim=2,missingslices=[],epsilonxy=0.0000025, epsilontheta=0.00000055,minepsilonxy=float(1e-20),minepsilontheta=2.0*10**-24,sigmaxy = 0.7*10, sigmatheta = np.pi/180*2*10, sigma_atlas = 1.0, sigma_target = 1.0,sigma_target_radius = 30,min_sigma_target = 1.0,sigma_atlas_radius = 34,min_sigma_atlas = 2.,norm=0,smoothing=1,optimizer='gd',beta_1=0.9,beta_2=0.999,epsilon_adam=10e-8,alpha_adam=0.001,sg_mask_mode = 'ones',sg_sigma=25):
        # equivalent of cor_sag_rot
        if dim != 2:
            target = torch.transpose(input,dim, 2).clone()
            input = torch.transpose(input,dim, 2).clone()
            template = torch.transpose(atlas,dim, 2).clone()
            if dim == 0:
                if dx == None:
                    dx = [self.dx[i] for i in [2, 1, 0]]
                else:
                    if len(dx) != 3:
                        print('ERROR: dx is not a list of length 3.')
                        return -1
                    else:
                        dx = [dx[i] for i in [2,1,0]]
                
                if nx == None:
                    nx = [self.nx[i] for i in [2, 1, 0]]
                else:
                    if len(nx) != 3:
                        print('ERROR: nx is not a list of length 3.')
                        return -1
                    else:
                        nx = [nx[i] for i in [2, 1, 0]]
            else:
                if dx == None:
                    dx = [self.dx[i] for i in [0, 2, 1]]
                else:
                    if len(dx) != 3:
                        print('ERROR: dx is not a list of length 3.')
                        return -1
                    else:
                        dx = [dx[i] for i in [0, 2, 1]]
                
                if nx == None:
                    nx = [self.nx[i] for i in [0, 2, 1]]
                else:
                    if len(nx) != 3:
                        print('ERROR: nx is not a list of length 3.')
                        return -1
                    else:
                        nx = [nx[i] for i in [0, 2, 1]]
        else:
            target = input.clone()
            template = atlas.clone()
            nx = self.nx
            dx = self.dx
        
        # allocate coordinate grid
        x0 = np.arange(nx[0])*dx[0]
        x1 = np.arange(nx[1])*dx[1]
        x2 = np.arange(nx[2])*dx[2]
        self.meanx0 = np.mean(x0)
        self.meanx1 = np.mean(x1)
        X0,X1 = np.meshgrid(x0-self.meanx0,x1-self.meanx1,indexing='ij')
        self.saX0 = torch.tensor(X0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.saX1 = torch.tensor(X1).type(self.params['dtype']).to(device=self.params['cuda'])
        X,Y,_ = np.meshgrid(x0-self.meanx0,x1-self.meanx1,x2-np.mean(x2),indexing='ij')
        X = torch.tensor(X).type(self.params['dtype']).to(device=self.params['cuda'])
        Y = torch.tensor(Y).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # allocate parameters
        if a is None:
            a = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        if b is None:
            b = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        if theta is None:
            theta = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # allocate stuff
        # need to account for missing slices here
        factor_vector = np.ones((nx[2]-len(missingslices),))
        factor_vector[0:sigma_target_radius] = np.linspace(min_sigma_target,1.,sigma_target_radius)
        factor_vector[-sigma_target_radius:] = np.linspace(1.,min_sigma_target,sigma_target_radius)
        sigma_target_vec = torch.tensor(np.ones((nx[2]-len(missingslices),)) * sigma_target * factor_vector).type(self.params['dtype']).to(device=self.params['cuda'])
        factor_vector = np.ones((nx[2]-len(missingslices),))
        factor_vector[0:sigma_atlas_radius] = np.linspace(min_sigma_atlas,1.,sigma_atlas_radius)
        factor_vector[-sigma_atlas_radius:] = np.linspace(1.,min_sigma_atlas,sigma_atlas_radius)
        sigma_atlas_vec = torch.tensor(np.ones((nx[2]-len(missingslices),)) * sigma_atlas * factor_vector).type(self.params['dtype']).to(device=self.params['cuda'])
        
        
        # get slice z coordinates
        #TODO: remove missing slices from the data
        slicenumbers = list(range(target.shape[2]))
        for i in range(len(missingslices)):
            slicenumbers.remove(missingslices[i])
        
        # make a version of the target with missing slices removed
        #target_rem = torch.gather(target, 2, torch.Tensor(np.array(slicenumbers).astype(np.int32)).to(device=self.params['cuda'])).clone()
        target_rem = target[:,:,slicenumbers].clone()
        
        slicecoord = torch.stack([torch.tensor((x+1.0)*dx[2] - dx[2]/2.0).type(self.params['dtype']).to(device=self.params['cuda']) for x in [slicenumbers[0]-2, slicenumbers[0]-1] + slicenumbers + [slicenumbers[-1]+1, slicenumbers[-1]+2]])
        shiftindices = [-2,-1,0,1,2]
        mynumerator = torch.tensor(np.zeros((len(shiftindices), slicecoord.shape[0]))).type(self.params['dtype']).to(device=self.params['cuda'])
        dz = ((torch.roll(slicecoord[2:-2],-1) - slicecoord[2:-2]) - (torch.roll(slicecoord[2:-2],1) - slicecoord[2:-2])) / 2.0
        # account for rollover
        dz[0] = dz[1]
        dz[-1] = dz[-2]
        
        # smooth?
        if smoothing == 1:
            mygauss = torch.tensor(mygaussian(1,9)).type(self.params['dtype']).to(device=self.params['cuda'])
            for i in range(input.shape[2]):
                input[:,:,i] = torch.squeeze(torch.nn.functional.conv2d(input[:,:,i].unsqueeze(0).unsqueeze(0),mygauss.unsqueeze(0).unsqueeze(0), stride=1, padding=4))
            for i in range(template.shape[2]):
                template[:,:,i] = torch.squeeze(torch.nn.functional.conv2d(template[:,:,i].unsqueeze(0).unsqueeze(0),mygauss.unsqueeze(0).unsqueeze(0), stride=1, padding=4))
        
        # downsample?
        
        # scale the image?
        if norm==1:
            (scalelist,_) = torch.topk(torch.flatten(input),100000)
            scale = scalelist[99999]
            print(scale)
            #template = template / scale
            target = target/scale
            input = input/scale
            #template = (template - torch.min(template)) / torch.max((template - torch.min(template)))
            #template = (template - torch.min(template)) / torch.max((template - torch.min(template)))
        else:
            scale = 1.0
        
        # optimizer
        if optimizer=="adam":
            best_E = torch.tensor(1e15).type(self.params['dtype']).to(device=self.params['cuda'])
            m_a = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            m_b = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            m_theta = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            v_a = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            v_b = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            v_theta = torch.zeros((nx[2],)).type(self.params['dtype']).to(device=self.params['cuda'])
            meshind_x = torch.linspace(0,target_rem.shape[0]-1,target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda'])
            meshind_y = torch.linspace(0,target_rem.shape[1]-1,target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda'])
            meshX,meshY = torch.meshgrid(meshind_x,meshind_y)

        
        # sgd mask
        M = torch.ones(target_rem.shape).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # loop
        iter = 1
        E = torch.tensor(1e15).type(self.params['dtype']).to(device=self.params['cuda'])
        while (iter < niter):
            # interpolate stack
            k = 0
            for i in range(nx[2]):
                if i in missingslices:
                    continue
                
                # maybe don't store TX and TY
                TX = torch.cos(theta[i])*self.saX1 + -1*torch.sin(theta[i])*self.saX0 + a[i]
                TY = torch.sin(theta[i])*self.saX1 + torch.cos(theta[i])*self.saX0 + b[i]
                #target[:,:,i] = torch.squeeze(grid_sample(torch.squeeze(torch.transpose(input.clone(),dim, 2)[:,:,i]/scale).unsqueeze(0).unsqueeze(0),torch.stack( ( (TX)/(nx[1]*dx[1]-dx[1])*2, (TY)/(nx[0]*dx[0]-dx[0])*2 ) ,dim=2).unsqueeze(0),padding_mode='border'))
                target[:,:,i] = torch.squeeze(grid_sample(torch.squeeze(input[:,:,i]).unsqueeze(0).unsqueeze(0),torch.stack( ( (TX)/(nx[1]*dx[1]-dx[1])*2, (TY)/(nx[0]*dx[0]-dx[0])*2 ) ,dim=2).unsqueeze(0),padding_mode='zeros'))
                target_rem[:,:,k] = target[:,:,i].clone()
                k += 1
            
            # centered difference approximate
            #DYTI = ( torch.cat( (target[1:,:,:], torch.zeros(1,nx[1],nx[2])), dim=0) - torch.cat( ( torch.zeros(1,nx[1],nx[2]), target[:-1,:,:]), dim=0) ) / dx[1] / 2.0
            #DXTI = ( torch.cat( (target[:,1:,:], torch.zeros(nx[1],1,nx[2])), dim=0) - torch.cat( ( torch.zeros(nx[1],1,nx[2]), target[:,:-1,:]), dim=0) ) / dx[0] / 2.0
            DXTI = (torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),-1,dims=1)[:,:,2:-2] - torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),1,dims=1)[:,:,2:-2]) / dx[0] / 2.0
            DYTI = (torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),-1,dims=0)[:,:,2:-2] - torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),1,dims=0)[:,:,2:-2]) / dx[1] / 2.0
            #DXTI[:,0,:] = target[:,1,:]
            #DXTI[:,-1,:] = -1*target[:,-2,:]
            #DYTI[0,:,:] = target[1,:,:]
            #DYTI[-1,:,:] = -1*target[-2,:,:]
            
            # 2nd derivative
            mynumerator *= 0.0 # hack to set to 0, not sure if actually faster
            for i in range(len(shiftindices)):
                myshiftind = list(shiftindices)
                myshiftind.remove(shiftindices[i])
                mynumerator[i,:] = 2.0 * torch.roll(slicecoord,myshiftind[0]) * torch.roll(slicecoord,myshiftind[1]) + 2.0 * torch.roll(slicecoord,myshiftind[0]) * torch.roll(slicecoord,myshiftind[2]) + 2.0 * torch.roll(slicecoord,myshiftind[0]) * torch.roll(slicecoord,myshiftind[3]) + 2.0 * torch.roll(slicecoord,myshiftind[1]) * torch.roll(slicecoord,myshiftind[2]) + 2.0 * torch.roll(slicecoord,myshiftind[1]) * torch.roll(slicecoord,myshiftind[3]) + 2.0 * torch.roll(slicecoord,myshiftind[2]) * torch.roll(slicecoord,myshiftind[3]) - 6.0 * slicecoord * torch.roll(slicecoord,myshiftind[0]) - 6.0 * slicecoord * torch.roll(slicecoord,myshiftind[1]) - 6.0 * slicecoord * torch.roll(slicecoord,myshiftind[2]) - 6.0 * slicecoord * torch.roll(slicecoord,myshiftind[3]) + 12.0 * slicecoord**2
            
            # this array is mirrored left-right for some reason
            # correct torch.roll(target) to account for zero padding
            LZTI = (torch.reshape( mynumerator[3,:] / ( (torch.roll(slicecoord,1) - slicecoord) * (torch.roll(slicecoord,1) - torch.roll(slicecoord,-1)) * (torch.roll(slicecoord,1) - torch.roll(slicecoord,2)) * (torch.roll(slicecoord,1) - torch.roll(slicecoord,-2)) ), [1,1,slicecoord.shape[0] ] ).expand([target_rem.shape[0], target_rem.shape[1], -1] ) * torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),1,dims=2))[:,:,2:-2] \
                + (torch.reshape( mynumerator[4,:] / ( (torch.roll(slicecoord,2) - slicecoord) * (torch.roll(slicecoord,2) - torch.roll(slicecoord,-1)) * (torch.roll(slicecoord,2) - torch.roll(slicecoord,-2)) * (torch.roll(slicecoord,2) - torch.roll(slicecoord,1)) ), [1,1,slicecoord.shape[0]]).expand( [target_rem.shape[0],target_rem.shape[1], -1] ) * torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),2,dims=2))[:,:,2:-2] \
                + (torch.reshape( mynumerator[1,:] / ( (torch.roll(slicecoord,-1) - slicecoord) * (torch.roll(slicecoord,-1) - torch.roll(slicecoord,1)) * (torch.roll(slicecoord,-1) - torch.roll(slicecoord,-2)) * (torch.roll(slicecoord,-1) - torch.roll(slicecoord,2)) ), [1,1,slicecoord.shape[0]]).expand( [target_rem.shape[0],target_rem.shape[1],-1] ) * torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),-1,dims=2))[:,:,2:-2] \
                + (torch.reshape( mynumerator[0,:] / ( (torch.roll(slicecoord,-2) - slicecoord) * (torch.roll(slicecoord,-2) - torch.roll(slicecoord,1)) * (torch.roll(slicecoord,-2) - torch.roll(slicecoord,2)) * (torch.roll(slicecoord,-2) - torch.roll(slicecoord,-1)) ), [1,1,slicecoord.shape[0]]).expand( [target_rem.shape[0],target_rem.shape[1],-1] ) * torch.roll(torch.nn.functional.pad(target_rem,(2,2),'constant',0),-2,dims=2))[:,:,2:-2] \
                + (torch.reshape( mynumerator[2,:] / ( (slicecoord - torch.roll(slicecoord,1)) * (slicecoord - torch.roll(slicecoord,2)) * (slicecoord - torch.roll(slicecoord,-1)) * (slicecoord - torch.roll(slicecoord,-2)) ), [1,1,slicecoord.shape[0]]).expand( [target_rem.shape[0],target_rem.shape[1],-1] ) * torch.nn.functional.pad(target_rem,(2,2),'constant',0) )[:,:,2:-2]
            
            # update sgd mask
            if optimizer=="adam" and sg_mask_mode=="gauss":
                center_x = (torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda'])
                center_y = (torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda'])
                M = mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,center_x,center_y)
                # normalize mask
                #M = M / (1/(2*sg_sigma**2))
                # make mask 3d
                M = M.unsqueeze(2).expand(-1,-1,target_rem.shape[2])
                #print(torch.max(M))
            elif optimizer=="adam" and sg_mask_mode=="2gauss":
                M = mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda'])) + mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda']))
                M = M.unsqueeze(2).expand(-1,-1,target_rem.shape[2])
            elif optimizer=="adam" and sg_mask_mode=="i2gauss":
                for i in range(M.shape[2]):
                    M[:,:,i] =  mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda'])) + mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda']))
            elif optimizer=="adam" and sg_mask_mode=="igauss":
                for i in range(M.shape[2]):
                    M[:,:,i] = mygaussian_torch_selectcenter_meshgrid(meshX,meshY,sg_sigma,(torch.rand(1)*target_rem.shape[0]).type(self.params['dtype']).to(device=self.params['cuda']),(torch.rand(1)*target_rem.shape[1]).type(self.params['dtype']).to(device=self.params['cuda']))
            elif optimizer=="adam" and sg_mask_mode=="rand":
                M = torch.rand(M.shape)
            
            # compute energies
            last_E = E.clone()
            Eimtarget = torch.sum(1.0 / torch.squeeze(sigma_target_vec)**2 * torch.squeeze(torch.sum(torch.sum(-LZTI * target_rem,dim=0),dim=0)) ) * np.prod(dx)/2.0
            Eimatlas = torch.sum(1.0 / sigma_atlas_vec**2 * torch.squeeze(torch.sum(torch.sum( (target_rem - template[:,:,slicenumbers])**2 , dim=0),dim=0))) * np.prod(dx)/2.0
            Eregxy = torch.sum((a[slicenumbers]**2 + b[slicenumbers]**2)*dz)/2.0/sigmaxy**2/slicecoord[2:-2].shape[0]
            Eregtheta = torch.sum((theta[slicenumbers]**2)*dz)/2.0/sigmatheta**2/slicecoord[2:-2].shape[0]
            E = Eimtarget + Eimatlas + Eregxy + Eregtheta
            
            # adjust optimization parameters
            # locked to gdr for now
            if optimizer == "gd":
                if E < last_E and iter > 1:
                    if np.mod(iter,2) == 0:
                        epsilonxy *= 1.04
                    else:
                        epsilontheta *= 1.04
                
                if E > last_E and iter > 1:
                    if np.mod(iter,2) == 0:
                        epsilonxy = epsilonxy/1.5
                        #print('reducing epsilonxy to ' + str(epsilonxy))
                        if epsilonxy < minepsilonxy and epsilontheta < minepsilontheta:
                            break
                    else:
                        epsilontheta = epsilontheta/1.5
                        #print('reducing epsilontheta to ' + str(epsilontheta))
                        if epsilontheta < minepsilontheta and epsilontheta < minepsilontheta:
                            break
            elif optimizer=="adam":
                if E < best_E:
                    best_E = E.clone()
                    best_a = a.clone()
                    best_b = b.clone()
                    best_theta = theta.clone()
            
            # print energies
            if iter > 1:
                start_time = end_time
            
            end_time = time.time()
            if iter > 1:
                print("iter: " + str(iter) + ", E= {:.3f}, Eim_t= {:.3f}, Eim_a= {:.3f}, ER_xy= {:.3f}, ER_t= {:.4f}, ep_xy= {:.4f}, ep_t= {:.4f}, time= {:.2f}.".format(E.item(),Eimtarget.item(),Eimatlas.item(),Eregxy.item(),Eregtheta.item(), epsilonxy, epsilontheta,end_time-start_time))
            else:
                print("iter: " + str(iter) + ", E= {:.3f}, Eim_t= {:.3f}, Eim_a= {:.3f}, ER_xy= {:.3f}, ER_t= {:.4f}, ep_xy= {:.4f}, ep_t= {:.4f}.".format(E.item(),Eimtarget.item(),Eimatlas.item(),Eregxy.item(),Eregtheta.item(), epsilonxy, epsilontheta))
            #print('iter: ' + str(iter) +', cost: '+ str(E) +', im_t = ' +str(Eimtarget)+ ', im_a = ' + str(Eimatlas) + ', regxy = ' +str(Eregxy)+ ', regt = ' +str(Eregtheta)+ ', ep = ' +str(epsilonxy)+ ', ' +str(epsilontheta))
            
            # compute gradients
            if np.mod(iter,2) == 1:
                gradx = 1/sigma_target_vec**2 * torch.squeeze(torch.sum(torch.sum(-2*M*LZTI*DXTI,dim=0),dim=0) * dx[0] * dx[1])
                grady = 1/sigma_target_vec**2 * torch.squeeze(torch.sum(torch.sum(-2*M*LZTI*DYTI,dim=0),dim=0) * dx[0] * dx[1])
            
            #diffatlas = torch.zeros(DXTI.shape)
            diffatlas = target_rem - template[:,:,slicenumbers]
            # set missing slices to zero
            #for i in missingslices:
            #    diffatlas[:,:,i] = torch.zeros((nx[0],nx[1],1))
            
            if np.mod(iter,2) == 1:
                gradx = gradx + 1/sigma_atlas_vec**2 * torch.squeeze(torch.sum(torch.sum(2*M*DXTI*diffatlas,dim=0),dim=0) * dx[0] * dx[1])
                grady = grady + 1/sigma_atlas_vec**2 * torch.squeeze(torch.sum(torch.sum(2*M*DYTI*diffatlas,dim=0),dim=0) * dx[0] * dx[1])
                gradx += torch.squeeze(a[slicenumbers])/sigmaxy**2
                grady += torch.squeeze(b[slicenumbers])/sigmaxy**2
            
            if np.mod(iter,2) == 0:
                gradtheta = 1/sigma_target_vec**2 * torch.squeeze(torch.sum(torch.sum( -2*M*LZTI*(DXTI*-1*X[:,:,slicenumbers] + DYTI*Y[:,:,slicenumbers]),dim=0),dim=0) * dx[0] * dx[1])
                gradtheta = gradtheta + 1/sigma_atlas_vec**2 * torch.squeeze(torch.sum(torch.sum( 2*M* (DXTI*-1*X[:,:,slicenumbers] + DYTI*Y[:,:,slicenumbers]) * diffatlas,dim=0),dim=0) * dx[0] * dx[1])
                gradtheta += torch.squeeze(theta[slicenumbers])/sigmatheta**2
            
            # update parameters
            # don't need to calculate gradtheta and gradxy every iteration
            #TODO: separate normalization into update step only
            if np.mod(iter,2) == 1:
                if optimizer=="gd":
                    a[slicenumbers] = a[slicenumbers]-epsilonxy*gradx
                    b[slicenumbers] = b[slicenumbers]-epsilonxy*grady
                elif optimizer=="adam":
                    m_a[slicenumbers] = (beta_1*m_a[slicenumbers] + (1-beta_1)*gradx) / (1-beta_1**(iter+1))
                    m_b[slicenumbers] = (beta_1*m_b[slicenumbers] + (1-beta_1)*grady) / (1-beta_1**(iter+1))
                    v_a[slicenumbers] = (beta_2*v_a[slicenumbers] + (1-beta_2)*(gradx**2)) / (1-beta_2**(iter+1))
                    v_b[slicenumbers] = (beta_2*v_b[slicenumbers] + (1-beta_2)*(grady**2)) / (1-beta_2**(iter+1))
                    a[slicenumbers] = a[slicenumbers] - alpha_adam * m_a[slicenumbers] / (torch.sqrt(v_a[slicenumbers]) + epsilon_adam)
                    b[slicenumbers] = b[slicenumbers] - alpha_adam * m_b[slicenumbers] / (torch.sqrt(v_b[slicenumbers]) + epsilon_adam)
            else:
                if optimizer=="gd":
                    theta[slicenumbers] = theta[slicenumbers]-epsilontheta*gradtheta
                elif optimizer=="adam":
                    m_theta[slicenumbers] = (beta_1*m_theta[slicenumbers] + (1-beta_1)*gradtheta) / (1-beta_1**(iter+1))
                    v_theta[slicenumbers] = (beta_2*v_theta[slicenumbers] + (1-beta_2)*(gradtheta**2)) / (1-beta_2**(iter+1))
                    theta[slicenumbers] = theta[slicenumbers] - alpha_adam * m_theta[slicenumbers] / (torch.sqrt(v_theta[slicenumbers]) + epsilon_adam)
            
            iter += 1
        
        if optimizer != "adam":
            best_a = a.clone()
            best_b = b.clone()
            best_theta = theta.clone()
        
        return best_a, best_b, best_theta, target*scale, epsilonxy, epsilontheta
    
    # apply stack alignment transform
    def applySA(self, input, a, b, theta, dx=None, nx=None, dim=2):
        if dim != 2:
            target = torch.transpose(input,dim, 2).clone()
            out = torch.transpose(input,dim, 2).clone()
            if dim == 0:
                if dx == None:
                    dx = [self.dx[i] for i in [2, 1, 0]]
                else:
                    if len(dx) != 3:
                        print('ERROR: dx is not a list of length 3.')
                        return -1
                    else:
                        dx = [dx[i] for i in [2,1,0]]
                
                if nx == None:
                    nx = [self.nx[i] for i in [2, 1, 0]]
                else:
                    if len(nx) != 3:
                        print('ERROR: nx is not a list of length 3.')
                        return -1
                    else:
                        nx = [nx[i] for i in [2, 1, 0]]
            else:
                if dx == None:
                    dx = [self.dx[i] for i in [0, 2, 1]]
                else:
                    if len(dx) != 3:
                        print('ERROR: dx is not a list of length 3.')
                        return -1
                    else:
                        dx = [dx[i] for i in [0, 2, 1]]
                
                if nx == None:
                    nx = [self.nx[i] for i in [0, 2, 1]]
                else:
                    if len(nx) != 3:
                        print('ERROR: nx is not a list of length 3.')
                        return -1
                    else:
                        nx = [nx[i] for i in [0, 2, 1]]
        else:
            target = input.clone()
            out = input.clone()
            if nx == None:
                nx = self.nx
            elif len(nx) != 3:
                print('ERROR: nx is not a list of length 3.')
                return -1
            
            if dx == None:
                dx = self.dx
            elif len(dx) != 3:
                print('ERROR: dx is not a list of length 3.')
                return -1
        
        # allocate coordinate grid
        x0 = np.arange(nx[0])*dx[0]
        x1 = np.arange(nx[1])*dx[1]
        self.meanx0 = np.mean(x0)
        self.meanx1 = np.mean(x1)
        X0,X1 = np.meshgrid(x0-self.meanx0,x1-self.meanx1,indexing='ij')
        self.saX0 = torch.tensor(X0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.saX1 = torch.tensor(X1).type(self.params['dtype']).to(device=self.params['cuda'])
        
        for i in range(nx[2]):
            # maybe don't store TX and TY
            TX = torch.cos(theta[i])*self.saX1 + -1*torch.sin(theta[i])*self.saX0 + a[i]
            TY = torch.sin(theta[i])*self.saX1 + torch.cos(theta[i])*self.saX0 + b[i]
            #target[:,:,i] = torch.squeeze(grid_sample(torch.squeeze(torch.transpose(input.clone(),dim, 2)[:,:,i]/scale).unsqueeze(0).unsqueeze(0),torch.stack( ( (TX)/(nx[1]*dx[1]-dx[1])*2, (TY)/(nx[0]*dx[0]-dx[0])*2 ) ,dim=2).unsqueeze(0),padding_mode='border'))
            out[:,:,i] = torch.squeeze(grid_sample(torch.squeeze(target[:,:,i]).unsqueeze(0).unsqueeze(0),torch.stack( ( (TX)/(nx[1]*dx[1]-dx[1])*2, (TY)/(nx[0]*dx[0]-dx[0])*2 ) ,dim=2).unsqueeze(0),padding_mode='zeros'))
        
        return out
    
    # main loop
    def registration(self):
        for it in range(self.params['niter']):
            # deform images forward
            if self.J[0].dim() == 2:
                if self.params['low_memory'] < 1:
                    self.forwardDeformation2d()
                
                if self.params['cc'] == 1:
                    self.runContrastCorrection()
                
                # update weight estimation
                if self.params['we'] > 0 and np.mod(it,self.params['nMstep']) == 0:
                    self.computeWeightEstimation()
                
                if self.params['do_lddmm'] == 1:
                    ER = self.calculateRegularizationEnergyVt2d()
                else:
                    ER = torch.tensor(0.0).type(self.params['dtype'])
                
                if self.params['optimizer'] == 'sgd' or self.params['optimizer'] == 'adam' or self.params['optimizer'] == 'rmsprop':
                    self.updateSGDMask()
                
                lambda1,EM = self.calculateMatchingEnergyMSE2d()
            else:
                if self.params['low_memory'] < 1:
                    self.forwardDeformation()
                
                if self.params['cc'] == 1:
                    self.runContrastCorrection()
                
                # update weight estimation
                if self.params['we'] > 0 and np.mod(it,self.params['nMstep']) == 0:
                    self.computeWeightEstimation()
                
                if self.params['do_lddmm'] == 1:
                    ER = self.calculateRegularizationEnergyVt()
                else:
                    ER = torch.tensor(0.0).type(self.params['dtype'])
                
                if self.params['optimizer'] == 'sgd' or self.params['optimizer'] == 'adam' or self.params['optimizer'] == 'rmsprop':
                    self.updateSGDMask()
                
                lambda1,EM = self.calculateMatchingEnergyMSE()
            
            # save variables
            E = ER+EM
            self.EMAll.append(EM)
            self.ERAll.append(ER)
            self.EAll.append(E.item())
            if self.params['checkaffinestep']:
                self.EMAffineT.append(EM)
            if it == 0 and self.params['savebestv']:
                self.bestE = E.clone()
            
            # print function
            if it > 0:
                start_time = end_time
            else:
                total_time = 0.0
            
            end_time = time.time()
            
            if self.params['verbose'] == 1:
                if it > 0:
                    total_time += end_time-start_time
                    #print('iter: ' + str(it) + ', E = ' + str(E.item()) + ', ER = ' + str(ER.item()) + ', EM = ' + str(EM.item()) + ', ep = ' + str((self.GDBeta*self.params['epsilon']).item()) + ', time = ' + str(end_time-start_time) + '.')
                    if self.params['checkaffinestep'] == 1 and self.params['do_affine'] > 0:
                        print("iter: " + str(it) + ", E= {:.3f}, ER= {:.3f}, EM= {:.3f}, epd= {:.3f}, del_Ev= {:.4f}, del_El= {:.4f}, del_Et= {:.4f}, time= {:.2f}s.".format(E.item(),ER.item(),EM.item(),(self.GDBeta*self.params['epsilon']).item(),self.ERAll[-1] + self.EMDiffeo[-1] - self.EAll[-2], self.EMAffineR[-1] - self.EMDiffeo[-1], self.EMAffineT[-1] - self.EMAffineR[-1],end_time-start_time))
                    else:
                        print("iter: " + str(it) + ", E= {:.3f}, ER= {:.3f}, EM= {:.3f}, epd= {:.3f}, time= {:.2f}s.".format(E.item(),ER.item(),EM.item(),(self.GDBeta*self.params['epsilon']).item(),end_time-start_time))
                else:
                    #print('iter: ' + str(it) + ', E = ' + str(E.item()) + ', ER = ' + str(ER.item()) + ', EM = ' + str(EM.item()) + ', ep = ' + str((self.GDBeta*self.params['epsilon']).item()) + '.')
                    print("iter: " + str(it) + ", E = {:.4f}, ER = {:.4f}, EM = {:.4f}, epd = {:.6f}.".format(E.item(),ER.item(),EM.item(),(self.GDBeta*self.params['epsilon']).item()))
            
            # or (self.EAll[-1]/self.EAll[-2] < 1-self.params['minenergychange'] and self.EAll[-2]/self.EAll[-3] < 1-self.params['minenergychange'] and self.EAll[-3]/self.EAll[-4] < 1-self.params['minenergychange'] and self.EAll[-4]/self.EAll[-5] < 1-self.params['minenergychange'])
            if it == self.params['niter']-1 or ((self.params['do_lddmm'] == 0 or self.GDBeta < self.params['minbeta']) and (self.params['do_affine']==0 or (self.GDBetaAffineR < self.params['minbeta'] and self.GDBetaAffineT < self.params['minbeta']))) or self.EAll[-1]/self.EAll[self.params['energy_fraction_from']] <= self.params['energy_fraction']:
                if ((self.params['do_lddmm'] == 0 or self.GDBeta < self.params['minbeta']) and (self.params['do_affine']==0 or (self.GDBetaAffineR < self.params['minbeta'] and self.GDBetaAffineT < self.params['minbeta']))):
                    print('Early termination: Energy change threshold reached.')
                elif self.EAll[-1]/self.EAll[self.params['energy_fraction_from']] <= self.params['energy_fraction']:
                    print('Early termination: Minimum fraction of initial energy reached.')
                
                print('Total elapsed runtime: {:.2f} seconds.'.format(total_time))
                break
            
            del E, ER, EM
            
            # update step sizes
            if self.params['we'] == 0 or (self.params['we'] > 0 and np.mod(it,self.params['nMstep']) != 0):
                updateflag = self.updateGDLearningRate()
                # if asked for, recompute images
                if updateflag:
                    if self.J[0].dim() == 2:
                        if self.params['low_memory'] < 1:
                            self.forwardDeformation2d()
                        
                        lambda1,EM = self.calculateMatchingEnergyMSE2d()
                    else:
                        if self.params['low_memory'] < 1:
                            self.forwardDeformation()
                        
                        lambda1,EM = self.calculateMatchingEnergyMSE()
            
            # calculate affine gradient
            if self.params['do_affine'] == 1:
                if self.J[0].dim()==3:
                    self.calculateGradientA(self.affineA,lambda1)
                elif self.J[0].dim()==2:
                    self.calculateGradientA2d(self.affineA,lambda1)
            elif self.params['do_affine'] == 2:
                if self.J[0].dim()==3:
                    self.calculateGradientA(self.affineA,lambda1,mode='rigid')
                elif self.J[0].dim()==2:
                    self.calculateGradientA2d(self.affineA,lambda1,mode='rigid')
            
            if self.params['low_memory'] > 0:
                torch.cuda.empty_cache()
            
            # calculate and update gradients
            if self.params['do_lddmm'] == 1:
                self.calculateAndUpdateGradientsVt(lambda1,iter=it)
            
            del lambda1
            
            # update affine
            if self.params['do_affine'] > 0:
                if self.J[0].dim() == 3:
                    self.updateAffine()
                elif self.J[0].dim() == 2:
                    self.updateAffine2d()
            
            # update weight estimation
            if self.params['we'] > 0 and np.mod(it,self.params['nMstep']) == 0:
                #self.computeWeightEstimation()
                self.updateWeightEstimationConstants()
            
    
    # reset all transforms
    def resetTransforms(self):
        if hasattr(self, 'vt0'): # we never reset lddmm variables
            self.vt0 = []
            self.vt1 = []
            self.vt2 = []
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                self.vt2.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
        
        if hasattr(self,'affineA'): # we never automatically reset affine variables
            self.affineA = torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.lastaffineA = torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda'])
            self.gradA = torch.tensor(np.zeros((4,4))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        if hasattr(self,'ccIbar'): # we never reset cc variables
            self.ccIbar = []
            self.ccJbar = []
            self.ccVarI = []
            self.ccCovIJ = []
            for i in range(len(self.I)):
                self.ccIbar.append(0.0)
                self.ccJbar.append(0.0)
                self.ccVarI.append(1.0)
                self.ccCovIJ.append(1.0)
        
        # weight estimation variables
        if hasattr(self,'W'): # if number of channels changed, reset everything
            self.W = [[] for i in range(len(self.I))]
            self.we_C = [[] for i in range(len(self.I))]
            for i in range(self.params['we']):
                if i == 0: # first index is the matching channel, the rest is artifacts
                    for ii in self.params['we_channels']: # allocate space only for the desired channels
                        self.W[ii].append(torch.tensor(0.9*np.ones((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
                else:
                    for ii in self.params['we_channels']:
                        self.W[ii].append(torch.tensor(0.1*np.ones((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).to(device=self.params['cuda']))
                        self.we_C[ii].append(torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda']))
        
        # optimizer update variables
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.GDBetaAffineR = float(1.0)
        self.GDBetaAffineT = float(1.0)
        return
    
    # delete transforms to save memory (mainly for running slice alignment using same object)
    def delete(self):
        if hasattr(self,'vt0'):
            del self.vt0
        if hasattr(self,'vt1'):
            del self.vt1
        if hasattr(self,'vt2'):    
            del self.vt2
        if hasattr(self,'affineA'):
            del self.affineA
        if hasattr(self,'gradA'):
            del self.gradA
        if hasattr(self,'It'):
            del self.It
        if hasattr(self,'W'):
            del self.W
        if hasattr(self,'we_C'):
            del self.we_C
        if hasattr(self,'X0'):
            del self.X0
        if hasattr(self,'X1'):
            del self.X1
        if hasattr(self,'X2'):
            del self.X2
        if hasattr(self,'Khat'):
            del self.Khat
        if hasattr(self,'saX0'):
            del self.saX0
        if hasattr(self,'saX1'):
            del self.saX1

        self.initializer_flags['lddmm'] = 1
        self.initializer_flags['affine'] = 1
        self.initializer_flags['cc'] = 1
        self.initializer_flags['we'] = 1
        torch.cuda.empty_cache()
        return
    
    # parse input transforms
    def parseInputVTransforms(self,vt0,vt1,vt2):
        varlist = [vt0,vt1,vt2]
        namelist = ['vt0','vt1','vt2']
        for i in range(len(varlist)):
            if varlist[i] is not None:
                if not isinstance(varlist[i],list):
                    print('ERROR: input \'' + str(namelist[i]) + '\' must be a list.')
                    return -1
                else:
                    if len(varlist[i]) != len(self.vt0):
                        print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of length ' + str(len(self.vt0)) + ', length ' + str(len(varlist[i])) + ' was received.')
                        return -1
                    else:
                        for ii in range(len(varlist[i])):
                            if not isinstance(varlist[i][ii],(np.ndarray,torch.Tensor)):
                                print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of numpy.ndarray or torch.Tensor.')
                                return -1
                            elif not varlist[i][ii].shape == self.vt0[ii].shape:
                                print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of numpy.ndarray or torch.Tensor of shapes ' + str(list(self.vt0[ii].shape)) + ', shape ' + str(list(varlist[i][ii].shape)) + ' was received.')
                                return -1
                
                if i == 0:
                    self.vt0 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt0 assigned.')
                elif i == 1:
                    self.vt1 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt1 assigned.')
                elif i == 2:
                    self.vt2 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt2 assigned.')
            
        return 1
    
    # parse input affine transforms
    def parseInputATransforms(self,affineA):
        if affineA is not None:
            if not isinstance(affineA,(np.ndarray, torch.Tensor)):
                print('ERROR: input affineA must be of type numpy.ndarray or torch.Tensor.')
                return -1
            else:
                if not affineA.shape == self.affineA.shape:
                    print('ERROR: input affineA must be of shape ' + str(list(self.affineA.shape)) + ', received shape ' + str(list(affineA.shape)) + '.')
                    return -1
            
            self.affineA = torch.Tensor(affineA).type(self.params['dtype']).to(device=self.params['cuda'])
            print('Custom affineA assigned.')
        
        return 1
    
    # save transforms to numpy arrays
    def outputTransforms(self):
        if hasattr(self,'affineA') and hasattr(self,'vt0'):
            return [x.cpu().numpy() for x in self.vt0], [x.cpu().numpy() for x in self.vt1], [x.cpu().numpy() for x in self.vt2], self.affineA.cpu().numpy()
        elif hasattr(self,'affineA'):
            return self.affineA.cpu().numpy()
        elif hasattr(self,'vt0'):
            return [x.cpu().numpy() for x in self.vt0], [x.cpu().numpy() for x in self.vt1], [x.cpu().numpy() for x in self.vt2]
        else:
            print('ERROR: no LDDMM or linear transforms to output.')
    
    # output deformed template
    def outputDeformedTemplate(self):
        if self.params['low_memory'] == 0:
            return [x[-1].cpu().numpy() for x in self.It]
        else:
            return [(self.applyThisTransformNT(x)).cpu().numpy() for x in self.I]

    # save files to disk
    def saveOutputs(self, save_template=False):
        if save_template:
            for i in range(len(self.I)):
                outimg = nib.AnalyzeImage(self.It[i][-1].to('cpu').numpy(),None)
                outimg.header['pixdim'][1:4] = self.dx
                nib.save(outimg,self.params['outdir'] + 'deformed_template_ch' + str(i) + '.img')
    
    # load transforms from numpy arrays into object
    def loadTransforms(self,vt0=None, vt1=None, vt2=None, affineA=None):
        # check parameters
        flag = self._checkParameters()
        if flag==-1:
            print('ERROR: parameters did not check out.')
            return
        
        if self.initializer_flags['load'] == 1:
            # load images
            flag = self._load(self.params['template'],self.params['target'],self.params['costmask'])
            if flag==-1:
                print('ERROR: images did not load.')
                return
            
        # initialize initialize
        if self.J[0].dim() == 2:
            self.initializeVariables2d()
        else:
            if self.params['do_lddmm'] == 1:
                self.initializeVariables()
            else:
                self.initializeVariables()
        
        varlist = [vt0,vt1,vt2]
        namelist = ['vt0','vt1','vt2']
        for i in range(len(varlist)):
            if varlist[i] is not None:
                if not isinstance(varlist[i],list):
                    print('ERROR: input \'' + str(namelist[i]) + '\' must be a list.')
                    return -1
                else:
                    if len(varlist[i]) != len(self.vt0):
                        print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of length ' + str(len(self.vt0)) + ', length ' + str(len(varlist[i])) + ' was received.')
                        return -1
                    else:
                        for ii in range(len(varlist[i])):
                            if not isinstance(varlist[i][ii],(np.ndarray,torch.Tensor)):
                                print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of numpy.ndarray or torch.Tensor.')
                                return -1
                            elif not varlist[i][ii].shape == self.vt0[ii].shape:
                                print('ERROR: input \'' + str(namelist[i]) + '\' must be a list of numpy.ndarray or torch.Tensor of shapes ' + str(list(self.vt0[ii].shape)) + ', shape ' + str(list(varlist[i][ii].shape)) + ' was received.')
                                return -1
                
                if i == 0:
                    self.vt0 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt0 assigned.')
                elif i == 1:
                    self.vt1 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt1 assigned.')
                elif i == 2:
                    self.vt2 = torch.Tensor(varlist[i]).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('Custom vt2 assigned.')
        
        if affineA is not None:
            if not isinstance(affineA,(np.ndarray, torch.Tensor)):
                print('ERROR: input affineA must be of type numpy.ndarray or torch.Tensor.')
                return -1
            else:
                if not affineA.shape == self.affineA.shape:
                    print('ERROR: input affineA must be of shape ' + str(list(self.affineA.shape)) + ', received shape ' + str(list(affineA.shape)) + '.')
                    return -1
            
            self.affineA = torch.Tensor(affineA).type(self.params['dtype']).to(device=self.params['cuda'])
            print('Custom affineA assigned.')
        
        return 1
            
    
    # convenience function
    def run(self, restart=True, vt0=None, vt1=None, vt2=None, affineA=None, save_template=False):
        # check parameters
        flag = self._checkParameters()
        if flag==-1:
            print('ERROR: parameters did not check out.')
            return
        
        if self.initializer_flags['load'] == 1:
            # load images
            flag = self._load(self.params['template'],self.params['target'],self.params['costmask'])
            if flag==-1:
                print('ERROR: images did not load.')
                return
            
        # initialize initialize
        if self.J[0].dim() == 2:
            self.initializeVariables2d()
        else:
            if self.params['do_lddmm'] == 1:
                self.initializeVariables()
            else:
                self.initializeVariables()
        
        # check for initializing transforms
        flag = self.parseInputVTransforms(vt0,vt1,vt2)
        if flag == -1:
            print('ERROR: problem with input velocity fields.')
            return

        flag = self.parseInputATransforms(affineA)
        if flag == -1:
            print('ERROR: problem with input linear transforms.')
            return

        # initialize stuff for gradient function
        self._allocateGradientDivisors()
        
        # initialize kernels
        if self.params['do_lddmm'] == 1:
            if self.J[0].dim() == 2:
                self.initializeKernels2d()
            else:
                self.initializeKernels()
        
        # main loop
        self.registration()
        
        # update epsilon
        if self.params['update_epsilon'] == 1:
            self.updateEpsilonAfterRun()
        
        # save outputs
        self.saveOutputs(save_template=save_template)
        
    
