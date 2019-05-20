import torch
import torchvision.utils as vutils
import sys
import os
import numpy as np

def make_figure(save_path, W=None):
    if W is None:
        model           = torch.load(os.path.join(save_path, 'model.pth'))
        W               = model['W']

    vutils.save_image(
                W,
                os.path.join(save_path, 'weights.png'),
                nrow=int(np.floor(np.sqrt(W.size(1)))),
                padding=2,
                normalize=True,
            )

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python make_figure.py SAVE_PATH')
        exit(1)

    make_figure(sys.argv[1])
