import torch
import numpy as np
from torch.autograd import Variable
from olshausen_whiten import whiten

class DataLoader:
    def __init__(self, npy_name):
        imgs = np.load(npy_name)
        n_imgs, n_channels, sz, _ = imgs.shape

        self.imgs       = imgs
        self.n_imgs     = n_imgs
        self.n_channels = n_channels
        self.img_size   = sz

        self.img_list   = np.random.permutation(np.arange(n_imgs))
        self.next_id    = 0

    def next_batch(self, batch_size, patch_size):
        if self.next_id + batch_size < self.n_imgs:
            batch_ids       = self.img_list[self.next_id:self.next_id+batch_size]
            self.next_id   += batch_size
            last_batch      = False
        else:
            batch_ids       = self.img_list[self.next_id:]
            self.img_list   = np.random.permutation(np.arange(self.n_imgs))
            self.next_id    = 0
            last_batch      = True

        x = self.imgs[batch_ids, :, :, :]
        b = np.zeros((len(batch_ids), self.n_channels, patch_size, patch_size))
        p_indices = np.random.randint(self.img_size - patch_size, size=(len(batch_ids),2))

        for i in range(len(batch_ids)):
            xcoord     = p_indices[i,0]
            ycoord     = p_indices[i,1]
            batch_img  = x[i, :, xcoord:xcoord+patch_size, ycoord:ycoord+patch_size]
            if np.random.rand() > 0.5:
                batch_img = batch_img[:,::-1,:]
            if np.random.rand() > 0.5:
                batch_img = batch_img[:,:,::-1]

            batch_img   = batch_img - batch_img.mean()
            batch_img   = batch_img / np.sqrt((batch_img * batch_img).mean())
            batch_img   = whiten(batch_img.squeeze())[None, ...]
            batch_img   = batch_img - batch_img.mean()
            batch_img   = batch_img / np.sqrt((batch_img * batch_img).mean())
            batch_img   = np.sqrt(0.1) * batch_img
            b[i,:,:,:]  = batch_img
            
        X = torch.from_numpy(b).float()

        return X, last_batch
        
