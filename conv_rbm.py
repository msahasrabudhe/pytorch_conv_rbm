import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from expander import MatrixExpander


LOSS_NAMES                                  = ['recon', 'sparsity']

class ProbMaxPool(nn.Module):
    def __init__(self, mp):
        super(ProbMaxPool, self).__init__()
        self.mp                             = mp
        pool_filter                         = torch.FloatTensor(1, 1, self.mp, self.mp).fill_(1)
        self.register_buffer('pool_filter', pool_filter)
        return

    def pooler(self, H):
        batch_size, n_channels, rows, cols  = H.size()
        H_pooled                            = F.conv2d(H, self.pool_filter.expand(n_channels, n_channels, -1, -1), stride=self.mp, padding=0)
        return H_pooled

    def forward(self, H):
        batch_size, n_channels, rows, cols  = H.size()

        if not hasattr(self, 'expander'):
            self.expander                   = MatrixExpander((rows // self.mp, cols // self.mp), self.mp).cuda()
        
#        print('H size before exp (H): ', H.size())
        H_exp                               = torch.exp(H)

#        print('H size before pool (H_exp): ', H_exp.size())
        H_pooled                            = self.pooler(H_exp) 
#        print('H size after pool (H_pooled): ', H_pooled.size())
        H_pooled_ex                         = self.expander(H_pooled)
#        print('H_size after expand (H_pooled_ex): ', H_pooled_ex.size())

        H_probs                             = H_exp / (1 + H_pooled_ex)

        return H_probs


class ConvRBM(nn.Module):
    def __init__(self, options):
        super(ConvRBM, self).__init__()

        self.options                        = options
        self.ws                             = options.model.weight_size
        self.in_channels                    = options.model.channels
        self.out_channels                   = options.model.num_weights
        self.mp                             = options.model.pool_size
        self.sparsity                       = options.model.sparsity
        self.sigm                           = options.model.sigmoid
        self.reuse_vbias                    = options.model.use_vbias
        self.k_CD                           = options.model.k_CD

        self.prob_max_pool                  = ProbMaxPool(self.mp)

        self._loss_scaling_dict             = []
        for _loss_name in LOSS_NAMES:
            self._loss_scaling_dict[_loss_name] = getattr(options.training, 'scale_' + _loss_name, 0)

        W_data                              = 1e-4 * (torch.FloatTensor(out_channels, in_channels, ws, ws).random_()%1000)
        bh_data                             = 1e-4 * (torch.FloatTensor(out_channels).random_()%1000)/1000.0
        bv_data                             = 1e-4 * (torch.FloatTensor(in_channels).random_()%1000)/1000.0

        self.W                              = nn.Parameter(data=W_data)
        self.bh                             = nn.Parameter(data=bh_data)
        self.bv                             = nn.Parameter(data=bv_data)

    def forward_conv(self, X):
        H                                   = F.conv2d(X, self.W, bias=self.bh, stride=1, padding=0)
        return H

    def backward_conv(self, H):
        pad                                 = self.ws - 1
        bias                                = self.bv if self.reuse_vbias else None
        Xhat                                = F.conv2d(H, self.W.transpose(0,1).transpose(2,3), bias=bias, stride=1, padding=pad)
        return Xhat

    def CD(self, X):
        self.H                              = self.forward_conv(X)
        self.Hprobs                         = self.prob_max_pool(self.H)
        self.Xhat                           = self.backward_conv(self.Hprobs)

        return self.Xhat

    def forward(self, X):
        self.X                              = X
        for k in range(self.k_CD):
            if k == 0:
                self.CD(X)
                self.Hprobs0                = self.Hprobs.clone()
            else:
                self.CD(self.Xhat)
        # Final forward pass, get H from X
        self.H                              = self.forward_conv(self.Xhat)
        self.Hprobs                         = self.prob_max_pool(self.H)

        return self.Xhat

    def compute_updates(self):
        batch_size                          = self.Hprobs.size(0)
        n_hidden                            = self.Hprobs.size(2) * self.Hprobs.size(3)

        for w in range(self.out_channels):
            for bs in range(batch_size):
                __conv                      = F.conv2d(self.X[bs:bs+1,:,:,:], self.Hprobs0[bs:bs+1,w:w+1,:,:].transpose(2,3), stride=1, padding=0)
                if bs == 0:
                    _conv                   = __conv
                else:
                    _conv                   = _conv + __conv
            
            _conv                           = 1./batch_size * _conv

            if w == 0:
                W_grad                      = _conv
            else:
                W_grad                      = torch.cat((W_grad, _conv), dim=0)
        self.W.grad                         = 1./n_hidden * 1./batch_size * W_grad

        dbiash                              = (self.Hprobs0 - self.Hprobs).view(batch_size,-1,n_hidden).mean(dim=2).mean(dim=0) 
        sparsity                            = self.sparsity - self.Hprobs.view(batch_size,-1,n_hidden).mean(dim=2).mean(dim=0)
        self.bh.grad                        = self._loss_scaling_dict['sparsity'] * (dbiash + sparsity)

        if self.reuse_vbias:
            n_visible                       = self.X.size(2) * self.X.size(3)
            dbiasv                          = (self.Xhat - self.X).view(batch_size,-1,n_visible).mean(dim=2).mean(dim=0)
            self.bv.grad                    = dbiasv

        return

    def compute_losses(self):
        self.loss_recon                     = self._loss_scaling_dict['recon'] * torch.mean((self.X - self.Xhat) ** 2)
        self.loss_sparsity                  = self._loss_scaling_dict['sparsity'] * (self.sparsity - torch.mean(self.Hprobs)) ** 2

        self.loss_total                     = self.loss_recon + self.loss_sparsity
        return self.loss_total

