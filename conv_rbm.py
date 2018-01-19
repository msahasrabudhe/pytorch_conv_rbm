import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ProbMaxPool:
    def __init__(self, mp):
        self.mp = mp
        return

    def get_probs(self, H):
        batch_size, n_channels, rows, cols = H.size()

        filt = torch.FloatTensor(self.mp,self.mp).fill_(1).cuda()
        filt = filt.view(1, 1, self.mp, self.mp)
        fv   = Variable(filt, requires_grad=False)

        H_e  = torch.exp(H)
        rv   = 1 + torch.cat([F.conv2d(H[:,i,:,:].unsqueeze(1), fv, stride=self.mp) for i in range(H.size(1))], 1)

        H_se = rv.unsqueeze(4).repeat(1,1,1,1,self.mp*self.mp)
        H_se = H_se.view(batch_size, n_channels, rows/self.mp, self.mp*self.mp, cols/self.mp)
        H_se = H_se.transpose(3,4).contiguous()
        H_se = H_se.view(batch_size, n_channels, rows, cols)

        H_probs = H_e/H_se        
        return H_probs

    def pool_from_probs(self, H_probs):
        pass


class ConvRBM(nn.Module):
    def __init__(self, ws, in_channels, out_channels, mp, sparsity, sigm=False, reuse_vbias=False, k_CD=1):
        super(ConvRBM, self).__init__()

        self.ws             = ws
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.mp             = mp    
        self.sparsity       = sparsity
        self.sigm           = sigm
        self.reuse_vbias    = reuse_vbias
        self.k_CD           = 1
   
        self.dW_prev        = torch.FloatTensor(in_channels, out_channels, ws, ws).fill_(0)
        self.db_prev        = torch.FloatTensor(out_channels).fill_(0)

        self.forward_conv   = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, ws, 1, 0, bias=True)
        )
        self.backward_conv       = nn.Sequential(
                nn.Conv2d(out_channels, in_channels, ws, 1, ws-1, bias=True)
        )
        self.prob_max_pool  = ProbMaxPool(mp)
        return

    def forward_pass(self, X=None):
        if X is None:
            self.hid_in = self.forward_conv(self.vis_probs)
        else:
            self.hid_in = self.forward_conv(X)
        return

    def backward_pass(self):
        self.vis_probs = self.backward_conv(self.hid_probs)
        if self.sigm:
            self.vis_probs = F.sigmoid(self.vis_probs)
        return

    def copy_transposed_weights(self):
        W = self.forward_conv.state_dict()['0.weight'].transpose(0,1).transpose(2,3).contiguous()
        self.backward_conv.state_dict()['0.weight'] = W
        return

    def CD(self):
        self.copy_transposed_weights()
        self.backward_pass() 

        self.forward_pass()
        self.hid_probs = self.prob_max_pool.get_probs(self.hid_in)


    def step(self, X):
        self.X          = X
        self.forward_pass(X=X)
        self.hid_probs  = self.prob_max_pool.get_probs(self.hid_in)
        self.hid_probs0 = self.hid_probs.clone()
    
        for k in range(self.k_CD):
            self.CD()

    def update(self, lr, momentum=0, l2_reg=0.1, lr_sparse=0.1):
        n_hid_units = self.hid_probs.size(2)*self.hid_probs.size(3)
        n_vis_units = self.vis_probs.size(2)*self.vis_probs.size(3)

        H_se_T  = self.hid_probs.transpose(2,3)
        H_se_T0 = self.hid_probs0.transpose(2,3)

        W       = self.forward_conv.state_dict()['0.weight']

        deltaW  = torch.FloatTensor(W.size())

        batch_size = self.X.size(0)

        for c in range(self.in_channels):
            for i in range(batch_size):
                deltaW[c,:,:,:] = 1.0/n_hid_units*(F.conv2d(self.X[i,:,:,:], H_se_T0) - F.conv2d(self.vis_probs[i,:,:,:], H_se_T[i,))   # FIX
            deltaW[c,:,:,:]     = 1.0/batch_size*deltaW
            deltaW[c,:,:,:]    += momentum*(self.dW_prev[c,:,:,:])
            deltaW[c,:,:,:]    += l2_reg*W[c,:,:,:]


        db_sparse   = self.sparsity - torch.mean(torch.sum(torch.sum(self.hid_probs, axis=3), axis=2), axis=0)
        db          = torch.mean(torch.sum(torch.sum(self.hid_probs0 - self.hid_probs, axis=3), axis=2), axis=0)  # Possible optimisation here. 
        db          = 1.0/n_hid_units*db + lr_sparse*db_sparse
        db         += momentum*(self.db_prev)

        # We do not update the visible biases for now. 
        self.dW_prev = deltaW
        self.db_prev = db

        # Update. 
        self.forward_conv.state_dict()['0.weight'] += lr*deltaW
        self.forward_conv.state_dict()['0.bias']   += lr*db
