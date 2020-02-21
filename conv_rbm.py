import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from expander import MatrixExpander

from IPython import embed

LOSS_NAMES                                  = ['recon', 'sparsity']

class ProbMaxPool(nn.Module):
    def __init__(self, p_size, mp):
        super(ProbMaxPool, self).__init__()
        self.mp                             = mp
        pool_filter                         = torch.FloatTensor(1, 1, self.mp, self.mp).fill_(1)
        self.max_pooler                     = nn.MaxPool2d(self.mp)
        self.expander                       = MatrixExpander((p_size[0] // self.mp, p_size[1] // self.mp), self.mp)
        self.register_buffer('pool_filter', pool_filter)
        return

    def pooler(self, H):
        batch_size, n_channels, rows, cols  = H.size()
        H_pooled                            = F.conv2d(H, self.pool_filter.expand(n_channels, n_channels, -1, -1), stride=self.mp, padding=0)
        return H_pooled

    def get_states(self, probs):
        mlim                                = 10000

        batch_size, n_channels, rows, cols  = probs.size()
        self.states                         = torch.FloatTensor(probs.size()).fill_(0).cuda()

        self.bc_pairs                       = torch.LongTensor([[b,c] for b in range(batch_size) for c in range(n_channels)]).cuda()

        self.this_state                     = torch.FloatTensor(batch_size, n_channels, self.mp * self.mp + 1).fill_(0).cuda()
        self.choice                         = torch.FloatTensor(batch_size, n_channels, 1).fill_(0).cuda()
        for r in range(0, rows, self.mp):
            for c in range(0, cols, self.mp):
                self.this_state.fill_(0)
                blocks                      = probs[:,:,r:r+self.mp,c:c+self.mp].contiguous().view(batch_size, n_channels, -1)
                blocks                      = torch.cat((blocks, 1 - blocks.sum(dim=-1, keepdim=True)), dim=-1)
                cumsum_probs                = torch.cumsum(blocks, dim=2)
                self.choice.random_(0, mlim+1)
                self.choice                 = 1./mlim * self.choice
                diff                        = cumsum_probs - self.choice
                diff[diff < 0]              = 5
                active                      = torch.argmin(diff, dim=-1).view(-1).contiguous().long()
                self.this_state[self.bc_pairs[:,0], self.bc_pairs[:,1], active] = 1
                self.states[:,:,r:r+self.mp,c:c+self.mp] = self.this_state[:,:,:self.mp*self.mp].view(batch_size,n_channels,self.mp,self.mp).contiguous()
        return self.states


    def forward(self, H):
        batch_size, n_channels, rows, cols  = H.size()

        # Make numerically more stable:: subtract maximum value from each block
        H_max_pooled                        = self.max_pooler(H)
        H_mp_ex                             = self.expander(H_max_pooled)
        H                                   = H - H_mp_ex

        # We need to add this value in the denominator instead of 1, because of 
        #   we removed the maximum for numerical stability.
        H_m_hmax                            = torch.exp(-1 * H_mp_ex)
        
        H_exp                               = torch.exp(H)

        H_pooled                            = self.pooler(H_exp) 
        H_pooled_ex                         = self.expander(H_pooled)

        H_probs                             = H_exp / (H_m_hmax + H_pooled_ex)

        H_states                            = self.get_states(H_probs)


        return H_probs, H_states


class ConvRBM(nn.Module):
    def __init__(self, options):
        super(ConvRBM, self).__init__()

        self.options                        = options
        self.ws                             = options.model.weight_size
        self.register_buffer('rev_idx_w', torch.arange(self.ws-1, -1, -1).long())
        self.nw_vis                         = options.model.channels
        self.nw_hid                         = options.model.num_weights
        self.mp                             = options.model.pool_size
        self.sparsity                       = options.model.sparsity
        self.sigm                           = options.model.sigmoid
        self.use_vbias                      = options.model.use_vbias
        self.k_CD                           = options.model.k_CD

        self.sigma_start                    = options.training.sigma_start
        self.sigma_stop                     = options.training.sigma_stop
        self.std_gaussian                   = self.sigma_start

        self.mom_init                       = options.training.mom_init
        self.mom_final                      = options.training.mom_final
        self.momentum                       = self.mom_init
        self.change_momentum                = options.training.change_momentum

        self.base_lr                        = options.optimiser.lr
        self.lr_decay_frac                  = 1.0
        self.lr                             = options.optimiser.lr
        self.lr_decay                       = options.optimiser.lr_decay
        self.lr_decay_step                  = options.optimiser.lr_decay_step

        h_shape                             = options.training.patch_size - self.ws + 1
        self.register_buffer('rev_idx_h', torch.arange(h_shape-1, -1, -1).long())
        self.prob_max_pool                  = ProbMaxPool((h_shape, h_shape), self.mp)

        self._loss_scaling_dict             = {}
        for _loss_name in LOSS_NAMES:
            self._loss_scaling_dict[_loss_name] = getattr(options.training, 'scale_' + _loss_name, 0)

        W_data                              = 1e-2 * torch.randn(self.nw_hid, self.nw_vis, self.ws, self.ws)
        bh_data                             = - 0.1 * torch.ones(self.nw_hid)
        bv_data                             = 0 * torch.randn(self.nw_vis)

        self.register_buffer('W', W_data)
        self.register_buffer('bh', bh_data)
        self.register_buffer('bv', bv_data)
        self.dummy                          = nn.Parameter(data=torch.FloatTensor(1).fill_(0))
#        self.W                              = nn.Parameter(data=W_data)
#        self.bh                             = nn.Parameter(data=bh_data)
#        self.bv                             = nn.Parameter(data=bv_data)

    def flip_updown(self, weight, h=False):
        if h:
            rev_idx                         = self.rev_idx_h
        else:
            rev_idx                         = self.rev_idx_w
        return weight[:, :, rev_idx, :][:, :, :, rev_idx]
# ===    return weight[:,:,self.rev_idx,:][:,:,:,self.rev_idx]

    def forward_conv(self, X):
#        for _w in range(self.nw_hid):
#            C                               = 1./(self.std_gaussian ** 2) * F.conv2d(X, self.W[[_w],:,:,:], bias=self.bh[_w:_w+1], stride=1, padding=0)
#            if _w == 0:
#                H                           = C
#            else:
#                H                           = torch.cat((H, C), dim=1)
        H                                   = 1./(self.std_gaussian ** 2) * F.conv2d(X, self.W, bias=self.bh, stride=1, padding=0)
#        H                                   = convolve(X, self.W, bias=self.bh)
#        H                                   = 1./(self.std_gaussian ** 2) * H
        return H

    def backward_conv(self, H):
        pad                                 = self.ws - 1
        bias                                = self.bv if self.use_vbias else None
#        for _w in range(self.nw_hid):
#            C                               = F.conv2d(H[:, [_w], :, :], self.flip_updown(self.W[[_w],:,:,:]), bias=None, stride=1, padding=pad)
#            if _w == 0:
#                Xhat                        = C
#            else:
#                Xhat                        = Xhat + C
        Xhat                                = F.conv2d(H, self.flip_updown(self.W.permute([1,0,2,3]), h=False), bias=bias, stride=1, padding=pad)
#        Xhat                                = convolve(H, self.W.transpose(0,1).transpose(2,3), bias=bias, padding=pad)
        return Xhat

    def fwbw(self, step=0):
        if step == 0:
            self.H0                         = self.forward_conv(self.X)
            self.Hprobs0, self.Hstates0     = self.prob_max_pool(self.H0)
            self.Xhat                       = self.backward_conv(self.Hstates0)
        else:
            self.H                          = self.forward_conv(self.Xhat)
            self.Hprobs, self.Hstates       = self.prob_max_pool(self.H)
            self.Xhat                       = self.backward_conv(self.Hstates)

        return

    def forward(self, X):
        self.X                              = X
        for k in range(self.k_CD):
            self.fwbw(step=k)

        # Final forward pass, get H from X
        self.H                              = self.forward_conv(self.Xhat)
        self.Hprobs, self.Hstates           = self.prob_max_pool(self.H)

        # Fix decay. 
        if self.std_gaussian > self.sigma_stop:
            self.std_gaussian               = self.std_gaussian * 0.99

        return self.Xhat

    def set_momentum(self, it):
        if it < self.change_momentum:
            self.momentum                   = self.mom_init
        else:
            self.momentum                   = self.mom_final

    def set_lr(self, epoch):
#        self.lr         = self.base_lr / (1 + self.lr_decay * epoch)
        if epoch in self.lr_decay_step:
            self.lr_decay_frac             *= self.lr_decay
            print('Decaying LR by a factor of %.4f' %(self.lr_decay))

        self.lr                             = self.base_lr * self.lr_decay_frac

    def compute_updates(self):
        batch_size                          = self.Hprobs.size(0)
        n_hidden                            = self.Hprobs.size(2) * self.Hprobs.size(3)

        if not hasattr(self, 'dW'):
            self.dW                         = 0     # Iteration 0
            self.dhbias                     = 0
            self.dvbias                     = 0

        for w in range(self.nw_hid):
            for bs in range(batch_size):
                __conv1                     = F.conv2d(self.X[bs:bs+1,:,:,:], self.Hprobs0[bs:bs+1,w:w+1,:,:], stride=1, padding=0)
                __conv2                     = F.conv2d(self.Xhat[bs:bs+1,:,:,:], self.Hprobs[bs:bs+1,w:w+1,:,:], stride=1, padding=0)
                if bs == 0:
                    _conv                   = __conv1 - __conv2
                else:
                    _conv                   = _conv + __conv1 - __conv2
            
            _conv                           = 1./batch_size * _conv

            if w == 0:
                W_grad                      = _conv
            else:
                W_grad                      = torch.cat((W_grad, _conv), dim=0)
        self.W.grad                         = self.lr * ( self._loss_scaling_dict['recon'] * (1./n_hidden * W_grad) - self.options.training.weight_decay * self.W ) + self.momentum * self.dW 
        self.dW                             = self.W.grad

        dbiash                              = self._loss_scaling_dict['recon'] * (self.Hprobs0 - self.Hprobs).view(batch_size,-1,n_hidden).mean(dim=2).mean(dim=0) 
        sparsity                            = self._loss_scaling_dict['sparsity'] * ( self.sparsity - self.Hprobs0.view(batch_size,-1,n_hidden).mean(dim=2).mean(dim=0) )
        self.bh.grad                        = self.lr * (  (dbiash + sparsity) ) + self.momentum * self.dhbias
        self.dhbias                         = self.bh.grad

        if self.use_vbias:
            n_visible                       = self.X.size(2) * self.X.size(3)
            dbiasv                          = (self.Xhat - self.X).view(batch_size,-1,n_visible).mean(dim=2).mean(dim=0)
            self.bv.grad                    = self.lr * ( dbiasv ) + self.momentum * self.dvbias 
            self.dvbias                     = self.bv.grad

        return

    def reset_grad(self):
        param_list                          = [self.W, self.bh]
        if self.use_vbias:
            param_list.append(self.bv)

        for param in param_list:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        return

    def update(self):

        self.W.add_(self.dW)
        self.bh.add_(self.dhbias)
        if self.use_vbias:
            self.bv.add_(self.dvbias)
            
#        param_list                          = [self.W, self.bh]
#        if self.use_vbias:
#            param_list.append(self.bv)
#        
#        for param in param_list:
#            param.data.add_(-1 * self.lr * param.grad)
#        return


    def compute_losses(self):
        self.loss_recon                     = self._loss_scaling_dict['recon'] * torch.mean((self.X - self.Xhat) ** 2)
        self.loss_sparsity                  = self._loss_scaling_dict['sparsity'] * torch.mean((self.sparsity - self.Hprobs.mean(dim=-1).mean(dim=-1)) ** 2)

        self.loss_total                     = self.loss_recon + self.loss_sparsity
        return self.loss_total


def convolve(image, filt, bias=None, stride=1, padding=0):
    batch_size                              = image.size(0)
    num_channels                            = image.size(1)
    num_weights                             = filt.size(0)
    assert(num_channels == filt.size(1))

    for w in range(num_weights):
        biasw                               = bias[w:w+1] if bias is not None else None
        for c in range(num_channels):
            if w == 0:
                _conv                       = F.conv2d(image[:,c:c+1,:,:], filt[w:w+1,c:c+1,:,:], bias=biasw, stride=stride, padding=padding)
            else:
                _conv                       = _conv + F.conv2d(image[:,c:c+1,:,:], filt[w:w+1,c:c+1,:,:], bias=biasw, stride=stride, padding=padding)
        if w == 0:
            result                          = _conv
        else:
            result                          = torch.cat((result, _conv), dim=1)
    return result
