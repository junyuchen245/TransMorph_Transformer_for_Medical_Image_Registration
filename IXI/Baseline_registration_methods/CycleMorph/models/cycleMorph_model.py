'''
CycleMorph

Original code retrieved from:
https://github.com/jongcye/MEDIA_CycleMorph

Original paper:
Kim, B., Kim, D. H., Park, S. H., Kim, J., Lee, J. G., & Ye, J. C. (2021).
CycleMorph: Cycle consistent unsupervised deformable image registration.
Medical Image Analysis, 71, 102036.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch, sys
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
from . import networks
from .loss import crossCorrelation3D, gradientLoss, NCC_vxm, Grad3d_vxm
import torch.nn as nn
from . import configs as configs
import matplotlib.pyplot as plt
#from ptflops import get_model_complexity_info

class cycleMorph(BaseModel):
    def name(self):
        return 'cycleMorph'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize[0]
        self.input_A = self.Tensor(nb, 1, size, size)
        self.input_B = self.Tensor(nb, 1, size, size)

        # load/define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.encoder_nc, opt.decoder_nc, opt.which_model_net, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.encoder_nc, opt.decoder_nc, opt.which_model_net, opt.init_type, self.gpu_ids)

        #macsG_A, paramsG_A = get_model_complexity_info(self.netG_A, input_res=(2, 160, 192, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
        #macsG_B, paramsG_B = get_model_complexity_info(self.netG_B, input_res=(2, 160, 192, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
        #print('{:<30}  {:<8}'.format('Computational complexity: ', macsG_A))
        #print('{:<30}  {:<8}'.format('Number of parameters: ', paramsG_A))
        #print('{:<30}  {:<8}'.format('Computational complexity: ', macsG_B))
        #print('{:<30}  {:<8}'.format('Number of parameters: ', paramsG_B))
        #sys.exit(0)

        if opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
        if not self.isTrain:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr

            # define loss functions
            self.criterionL2 = Grad3d_vxm('l2')
            self.criterionCC = NCC_vxm()#crossCorrelation3D(1, kernel=(9,9,9))
            self.criterionCy = torch.nn.L1Loss()
            self.criterionId = NCC_vxm()#crossCorrelation3D(1, kernel=(9,9,9))

            # initialize optimizers
            self.optimizer_ = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input[0]
        input_B = input[1]
        self.input_A = input_A
        self.input_B = input_B
        #self.input_A.resize_(input_A.size()).copy_(input_A)
        #self.input_B.resize_(input_B.size()).copy_(input_B)
        #self.image_paths = input['path']

    def forward(self):
        self.real_A = self.input_A#Variable(self.input_A)
        self.real_B = self.input_B#Variable(self.input_B)

    def test(self):
        self.netG_A.eval()
        with torch.no_grad():
            real_A = Variable(self.input_A)
            real_B = Variable(self.input_B)
            _, flow_A = self.netG_A(torch.cat((real_A, real_B), dim=1).to(torch.device("cuda")))
        self.flow_A = flow_A 

    #def get_image_paths(self):
    #    return self.image_paths

    def backward_G(self):
        self.netG_A.train()
        self.netG_B.train()
        lambda_ = self.opt.lambda_R
        alpha = self.opt.lambda_A
        beta = self.opt.lambda_B

        # Registration loss
        fake_B, flow_A = self.netG_A(torch.cat([self.real_A, self.real_B], dim=1))
        #fake_B_tmp = fake_B.detach().cpu().numpy()[0, 0, :, :, 122]
        #real_A_tmp = self.real_A.detach().cpu().numpy()[0, 0, :, :, 122]
        #real_B_tmp = self.real_B.detach().cpu().numpy()[0, 0, :, :, 122]
        #plt.figure()
        #plt.subplot(1, 3, 1)
        #plt.imshow(fake_B_tmp, cmap='gray')
        #plt.subplot(1, 3, 2)
        #plt.imshow(real_A_tmp, cmap='gray')
        #plt.subplot(1, 3, 3)
        #plt.imshow(real_B_tmp, cmap='gray')
        #plt.show()
        lossA_RC = self.criterionCC(fake_B, self.real_B)
        #print(lossA_RC)
        lossA_reg = self.criterionL2(flow_A)
        lossA_RL = lossA_reg * lambda_

        fake_A, flow_B = self.netG_B(torch.cat([self.real_B, self.real_A], dim=1))
        lossB_RC = self.criterionCC(fake_A, self.real_A)
        lossB_reg = self.criterionL2(flow_B)
        lossB_RL = lossB_reg * lambda_

        # Cycle loss
        back_A, bflow_A = self.netG_B(torch.cat([fake_B, fake_A], dim=1))
        lossA_CY = self.criterionCy(back_A, self.real_A) * alpha
        back_B, bflow_B = self.netG_A(torch.cat([fake_A, fake_B], dim=1))
        lossB_CY = self.criterionCy(back_B, self.real_B) * alpha

        # Identity loss
        idt_A, iflow_A = self.netG_A(torch.cat([self.real_B, self.real_B], dim=1))
        lossA_ID = self.criterionId(idt_A, self.real_B) * beta
        idt_B, iflow_B = self.netG_B(torch.cat([self.real_A, self.real_A], dim=1))
        lossB_ID = self.criterionId(idt_B, self.real_A) * beta

        loss = lossA_RC + lossA_RL + lossB_RC + lossB_RL + lossA_CY + lossB_CY + lossA_ID + lossB_ID
        loss.backward()

        self.flow_A  = flow_A.data
        self.flow_B  = flow_B.data
        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.back_A  = back_A.data
        self.back_B  = back_B.data
        self.lossA_RC = lossA_RC.item()
        self.lossA_RL = lossA_RL.item()
        self.lossB_RC = lossB_RC.item()
        self.lossB_RL = lossB_RL.item()
        self.lossA_CY = lossA_CY.item()
        self.lossB_CY = lossB_CY.item()
        self.lossA_ID = lossA_ID.item()
        self.lossB_ID = lossB_ID.item()

        self.loss    = loss.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_.zero_grad()
        self.backward_G()
        self.optimizer_.step()

        return self.loss, self.lossA_RL, self.lossA_RC

    def get_current_errors(self):
        ret_errors = OrderedDict([('A_RC', self.lossA_RC), ('A_RL', self.lossA_RL),
                                  ('B_RC', self.lossB_RC), ('B_RL', self.lossB_RL),
                                  ('A_CY', self.lossA_CY), ('B_CY', self.lossB_CY),
                                  ('A_ID', self.lossA_ID), ('B_ID', self.lossB_ID),
                                  ('Tot', self.loss)])
        return ret_errors

    def get_current_visuals(self):
        realSize = self.input_A.shape
        real_A = util.tensor2im(self.input_A[0, 0, int(realSize[2]/2)])
        flow_A = util.tensor2im(self.flow_A[0, :, int(realSize[2] / 2)])
        fake_B = util.tensor2im(self.fake_B[0, 0, int(realSize[2]/2)])
        back_A = util.tensor2im(self.back_A[0, 0, int(realSize[2] / 2)])

        real_B = util.tensor2im(self.input_B[0, 0, int(realSize[2]/2)])
        flow_B = util.tensor2im(self.flow_B[0, :, int(realSize[2]/2)])
        fake_A = util.tensor2im(self.fake_A[0, 0, int(realSize[2] / 2)])
        back_B = util.tensor2im(self.back_B[0, 0, int(realSize[2] / 2)])

        ret_visuals = OrderedDict([('real_A', real_A), ('flow_A', flow_A),
                                   ('fake_B', fake_B), ('back_A', back_A),
                                   ('real_B', real_B), ('flow_B', flow_B),
                                   ('fake_A', fake_A), ('back_B', back_B)])
        return ret_visuals

    def get_current_data(self):
        ret_visuals = OrderedDict([('flow_A', self.flow_A),('fake_B', self.fake_B)])
        return ret_visuals

    def get_test_data(self):
        ret_visuals = OrderedDict([('flow_A', self.flow_A)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)


CONFIGS = {
    'Cycle-Morph-v0': configs.get_CycleMorphV0_config(),
}

