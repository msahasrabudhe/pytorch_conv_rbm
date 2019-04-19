import torch
import numpy as np
from torch.autograd import Variable

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
        else:
            batch_ids       = self.img_list[self.next_id:]
            self.img_list   = np.random.permutation(np.arange(self.n_imgs))
            self.next_id    = 0

        x = self.imgs[batch_ids, :, :, :]
        b = np.zeros((len(batch_ids), self.n_channels, patch_size, patch_size))
        p_indices = np.random.randint(self.img_size - patch_size, size=(len(batch_ids),2))

        for i in range(len(batch_ids)):
            xcoord     = p_indices[i,0]
            ycoord     = p_indices[i,1]
            b[i,:,:,:] = x[i, :, xcoord:xcoord+patch_size, ycoord:ycoord+patch_size]

        X = torch.from_numpy(b).float()
        return X
        
