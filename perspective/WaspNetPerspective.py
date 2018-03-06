import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import numpy as np

from matplotlib.mlab import griddata
from matplotlib import tri

# Menpo stuff. 
from menpo.shape import TexturedTriMesh
from menpo3d.rasterize import rasterize_barycentric_coordinate_images
import menpo.image as mimage
import gc

def getBaseGrid(N=64, normalize = True, getbatch = False, batchSize = 1):
    a = torch.arange(-(N-1), (N), 2)
    if normalize:
        a = a/(N-1)
    x = a.repeat(N,1)
    y = x.t()
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)),0)
    if getbatch:
        grid = grid.unsqueeze(0).repeat(batchSize,1,1,1)
    return grid


# Returns a set of faces which are given by a triangulation of a
#   regular 2D grid of size (imgSize x imgSize)
def get_tri_list(imgSize):
    n_tri    = 2*(imgSize - 1)*(imgSize - 1)
    tri_list = np.zeros((n_tri, 3))

    tl_ids = np.array([i*imgSize + np.arange(imgSize - 1) for i in range(imgSize - 1)]).flatten()
    tl_id1, tl_id2, tl_id3 = tl_ids, tl_ids + 1, tl_ids + imgSize
    br_ids = np.array([i*imgSize + 1+np.arange(imgSize-1) for i in range(1,imgSize)]).flatten()
    br_id1, br_id2, br_id3 = br_ids, br_ids - 1, br_ids - imgSize
    l = br_ids.size
    f_tl_ids = np.array([tl_id1, tl_id2, tl_id3]).transpose()
    f_br_ids = np.array([br_id3, br_id2, br_id1]).transpose()
    tri_list[:l, :] = f_tl_ids
    tri_list[l:, :] = f_br_ids
    return tri_list.astype(np.int)


def my_inverse_warp(img, grid, opt):
    grid = (grid + 1)/2
    s    = grid.shape
            
    Ux, Uy = np.where(grid[0,:,:] <= 0)
    Vx, Vy = np.where(grid[1,:,:] <= 0)
    grid[0,Ux,Uy] = 0.0
    grid[1,Vx,Vy] = 0.0
                
    U    = grid[0,:,:].flatten()
    V    = grid[1,:,:].flatten()
        
    valuesR = img[0,:,:].flatten()
    valuesG = img[1,:,:].flatten()
    valuesB = img[2,:,:].flatten()
        
    num_indices = opt.imgSize
    xi   = np.linspace(0, 1.0, num_indices)
    yi   = np.linspace(0, 1.0, num_indices)

    Tex_R = np.array(griddata(U, V, valuesR, xi, yi, interp='linear')); Tex_R[np.isnan(Tex_R)] = 0
    Tex_G = np.array(griddata(U, V, valuesG, xi, yi, interp='linear')); Tex_G[np.isnan(Tex_G)] = 0
    Tex_B = np.array(griddata(U, V, valuesB, xi, yi, interp='linear')); Tex_B[np.isnan(Tex_B)] = 0
                    
    TEXTURE = np.concatenate((Tex_R[np.newaxis,:,:] ,Tex_G[np.newaxis,:,:],Tex_B[np.newaxis,:,:] ) , axis =0)
    return TEXTURE

def my_tri_inverse_warp(img, grid, opt):
    grid = (grid + 1)/2
    s    = grid.shape

    Ux, Uy = np.where(grid[0,:,:] <= 0)
    Vx, Vy = np.where(grid[1,:,:] <= 0)
    grid[0,Ux,Uy] = 0.0
    grid[1,Vx,Vy] = 0.0
    U = grid[0,:,:].flatten()
    V = grid[1,:,:].flatten()

    valuesR = img[0,:,:].flatten()
    valuesG = img[1,:,:].flatten()
    valuesB = img[2,:,:].flatten()

    num_indices = opt.imgSize
    xi = np.linspace(0, 1.0, num_indices)
    yi = np.linspace(0, 1.0, num_indices)
    xi, yi = np.meshgrid(xi, yi)

    triR = tri.Triangulation(U, V); texR = tri.LinearTriInterpolator(triR, valuesR)
    triB = tri.Triangulation(U, V); texB = tri.LinearTriInterpolator(triB, valuesB)
    triG = tri.Triangulation(U, V); texG = tri.LinearTriInterpolator(triG, valuesG)

    Tex_R = texR(xi,yi);   nanR = np.isnan(Tex_R);   Tex_R[nanR] = 0.0;
    Tex_B = texB(xi,yi);   nanB = np.isnan(Tex_B);   Tex_B[nanB] = 0.0;
    Tex_G = texG(xi,yi);   nanG = np.isnan(Tex_G);   Tex_G[nanG] = 0.0;

    texture = np.concatenate((Tex_R[np.newaxis,:,:], Tex_G[np.newaxis,:,:], Tex_B[np.newaxis,:,:]), axis=0)
    return texture


# ================================================================================= 
from menpo3d.rasterize.cpu import * 
# Overwrite xy_bcoords, alpha_beta, rasterize_barycentric_coordinates, and
#   rasterize_barycentric_coordinate_images so that we also get the value of d in alpha_beta. 
def my_alpha_beta(i, ij, ik, points):
    ip = points - i
    dot_jj = np.einsum('dt, dt -> t', ij, ij)
    dot_kk = np.einsum('dt, dt -> t', ik, ik)
    dot_jk = np.einsum('dt, dt -> t', ij, ik)
    dot_pj = np.einsum('dt, dt -> t', ip, ij)
    dot_pk = np.einsum('dt, dt -> t', ip, ik)

    d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk)
    alpha = (dot_kk * dot_pj - dot_jk * dot_pk) * d
    beta = (dot_jj * dot_pk - dot_jk * dot_pj) * d
    return alpha, beta, d

def my_xy_bcoords(mesh, tri_indices, pixel_locations):
    i, ij, ik = barycentric_vectors(mesh.points[:, :2], mesh.trilist)
    i = i[:, tri_indices]
    ij = ij[:, tri_indices]
    ik = ik[:, tri_indices]
    a, b, phi = my_alpha_beta(i, ij, ik, pixel_locations.T)
    c = 1 - a - b
    bcoords = np.array([c, a, b]).T
    return bcoords, phi

def my_rasterize_barycentric_coordinates(mesh, image_shape):
    height, width = int(image_shape[0]), int(image_shape[1])
    # 1. Find all pixel-sites that may need to be rendered to
    #    + the triangle that may partake in rendering
    yx, tri_indices = pixel_locations_and_tri_indices(mesh)

    # 2. Limit to only pixel sites in the image
    out_of_bounds = np.logical_or(
        np.any(yx < 0, axis=1),
        np.any((np.array([height, width]) - yx) <= 0, axis=1))
    in_image = ~out_of_bounds
    yx = yx[in_image]
    tri_indices = tri_indices[in_image]

    # # Optionally limit to subset of pixels
    # if n_random_samples is not None:
    #     # 2. Find the unique pixel sites
    #     xy_u = unique_locations(yx, width, height)
    #
    #     xy_u = pixel_sample_uniform(xy_u, n_random_samples)
    #     to_keep = np.in1d(location_to_index(yx, width),
    #                       location_to_index(xy_u, width))
    #     yx = yx[to_keep]
    #     tri_indices = tri_indices[to_keep]

    bcoords, phi = my_xy_bcoords(mesh, tri_indices, yx)

    # check the mask based on triangle containment
    in_tri_mask = tri_containment(bcoords)

    # use this mask on the pixels
    yx = yx[in_tri_mask]
    bcoords = bcoords[in_tri_mask]
    phi     = phi[in_tri_mask]
    tri_indices = tri_indices[in_tri_mask]

    # Find the z values for all pixels and calculate the mask
    z_values = z_values_for_bcoords(mesh, bcoords, tri_indices)

    # argsort z from smallest to biggest - use this to sort all data
    sort = np.argsort(z_values)
    yx = yx[sort]
    bcoords = bcoords[sort]
    phi     = phi[sort]
    tri_indices = tri_indices[sort]

    # make a unique id per-pixel location
    pixel_index = yx[:, 0] * width + yx[:, 1]
    # find the first instance of each pixel site by depth
    _, z_buffer_mask = np.unique(pixel_index, return_index=True)

    # mask the locations one last time
    yx = yx[z_buffer_mask]
    bcoords = bcoords[z_buffer_mask]
    phi     = phi[z_buffer_mask]
    tri_indices = tri_indices[z_buffer_mask]
    return yx, bcoords, tri_indices, phi

def my_rasterize_barycentric_coordinate_images(mesh, image_shape):
    h, w = image_shape
    yx, bcoords, tri_indices, phi = my_rasterize_barycentric_coordinates(mesh,
                                                                 image_shape)

    tri_indices_img = np.zeros((1, h, w), dtype=int)
    bcoords_img = np.zeros((3, h, w))
    phi_img = np.zeros((1, h, w))
    mask = np.zeros((h, w), dtype=np.bool)
    mask[yx[:, 0], yx[:, 1]] = True
    tri_indices_img[:, yx[:, 0], yx[:, 1]] = tri_indices
    bcoords_img[:, yx[:, 0], yx[:, 1]] = bcoords.T
    phi_img[:, yx[:, 0], yx[:, 1]] = phi

    mask = BooleanImage(mask)
    return (MaskedImage(bcoords_img, mask=mask.copy(), copy=False),
            MaskedImage(tri_indices_img, mask=mask.copy(), copy=False), phi_img)

# =================================== Finished. =================================== 
   
#    plt.figure()
#    TEXTURE = (TEXTURE - np.min(TEXTURE))/(np.max(TEXTURE) - np.min(TEXTURE))
#    plt.imshow(TEXTURE)
#    plt.show()

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target, mask):
        self.loss = self.criterion(torch.mul(input,mask), torch.mul(target,mask))
        return self.loss


class MaskedABSLoss(nn.Module):
    def __init__(self):
        super(MaskedABSLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):
        self.loss = self.criterion(torch.mul(input,mask), torch.mul(target,mask))
        return self.loss

# Linear, 2-layer regressor.
class waspLandmarkRegressorLinear(nn.Module):
    def __init__(self, opt):
        super(waspLandmarkRegressorLinear, self).__init__()
        self.ngpu    = opt.ngpu
        self.ndim    = 10
        self.imgSize = opt.imgSize
        self.nc      = 2
        self.main = nn.Sequential(
            nn.Linear(self.imgSize*self.imgSize*self.nc, self.ndim*10),
            nn.ReLU(),
            nn.Linear(self.ndim*10, self.ndim)
#            nn.Linear(self.imgSize*self.imgSize*self.nc, self.ndim)
        )

    def forward(self, input):
        output = self.main(input.view(-1, self.imgSize*self.imgSize*self.nc).contiguous())
        return output

# Regressor to regress landmark locations. 
class waspLandmarkRegressor(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf=32):
        super(waspLandmarkRegressor, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequenaitl(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size 
            nn.Conv2d(ndf * 8, 10, 4, 4, 0, bias=False),
        )

    def forward(self, input):
        output = self.main(input).view(-1, 10)
        return output

# an encoder architecture
class waspEncoder(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128):
        super(waspEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output   

# an encoder architecture
class waspEncoderInject(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32, ndim = 128, injdim = 10):
        super(waspEncoderInject, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim
        self.injdim = injdim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndim+injdim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim+self.injdim)
        #print(output.size())
        return output 

# an encoder architecture
class waspEncoderInject2(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32):
        super(waspEncoderInject2, self).__init__()
        self.opt = opt
        self.ngpu = ngpu
        self.injdim = opt.injdim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.opt.injdim + self.opt.zdim_inj + self.opt.zdim_inj, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.opt.injdim + self.opt.zdim_inj + self.opt.zdim_inj)
        #print(output.size())
        return output 

# an encoder architecture
class waspEncoderInject3(nn.Module):
    def __init__(self, opt, ngpu=1, nc=1, ndf = 32):
        super(waspEncoderInject3, self).__init__()
        self.opt = opt
        self.ngpu = ngpu
        self.injdim = opt.injdim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.opt.injdim + self.opt.idim + self.opt.wdim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input).view(-1,self.opt.injdim + self.opt.idim + self.opt.wdim)
        #print(output.size())
        return output 

# a mixer (linear layer)
class waspMixer(nn.Module):
    def __init__(self, opt, ngpu=1, nin=128, nout=128):
        super(waspMixer, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # simply a linear layer
            nn.Linear(nin, nout),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# A non-linear layer with full connections on top. 
class waspNonLinearity(nn.Module):
    def __init__(self, opt, ngpu=1, nin=128, nout=128):
        super(waspNonLinearity, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Just a non-linearity. 
            nn.Linear(nin, nout),
            nn.ReLU(),
            nn.Linear(nout, nout)
        )

    def forward(self, inputs):
        if self.ngpu > 1 and isinstance(inputs.data, torch.cuda.FloatTensor):
            outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            outputs = self.main(inputs)
        return outputs


# a mixer (linear layer)
class waspSlicer(nn.Module):
    def __init__(self, opt, ngpu=1, pstart = 0, pend=1):
        super(waspSlicer, self).__init__()
        self.ngpu = ngpu
        self.pstart = pstart
        self.pend = pend
    def forward(self, input):
        output = input[:,self.pstart:self.pend].contiguous()
        return output


# a decoder architecture
class waspDecoder(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderTextureCode(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTextureCode, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class FinalCodeDecoder(nn.Module):
    def __init__(self, opt, ngpu, ngf=3, nc=3, lb=0, ub=1):
        super(FinalCodeDecoder, self).__init__()
        self.opt  = opt
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(ngf, opt.tdf2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(opt.tdf2),
            nn.ReLU(True),

            nn.Conv2d(opt.tdf2, opt.nc, 3, 1, 1, bias=False)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# Decoder for affine transform. Just a linear layer. 
class waspDecoderLinear(nn.Module):
    def __init__(self, opt, ngpu=1, nz=10, ndim=6):
        super(waspDecoderLinear, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is z. Just apply a linear layer on top. 
            nn.Linear(nz, ndim),
            nn.ReLU(),
            nn.Linear(ndim, ndim)
        )
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))    
        else:
            output = self.main(input)
#        print(output.view(-1,2,3).size())
#        return output.contiguous().view(-1,2,3)
        return output
# Decoder for affine transform. Just a linear layer. 
class waspDecoderLinearSigmoid(nn.Module):
    def __init__(self, opt, ngpu=1, nz=10, ndim=6):
        super(waspDecoderLinearSigmoid, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is z. Just apply a linear layer on top. 
            nn.Linear(nz, ndim),
            nn.ReLU(),
            nn.Linear(ndim, ndim),
            nn.Sigmoid()
        )
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))    
        else:
            output = self.main(input)
#        print(output.view(-1,2,3).size())
#        return output.contiguous().view(-1,2,3)
        return output



# a decoder architecture
class waspDecoderSigm(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderSigm, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoder2(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder2, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# Decoder for Z. 
class waspDepthDecoderTanh(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1):
        super(waspDepthDecoderTanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution. 
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), 
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            outputs = self.main(inputs)
        return outputs

# a decoder architecture
class waspDecoderTanhNS(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTanhNS, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderTanh(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            #nn.Hardtanh(lb,ub),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderHardTanh(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderHardTanh, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub),
            #nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoder_B(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoder_B, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Hardtanh(lb,ub)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# a decoder architecture
class waspDecoderTanh_B(nn.Module):
    def __init__(self, opt, ngpu=1, nz=128,  nc=1, ngf=32, lb=0, ub=1):
        super(waspDecoderTanh_B, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            #nn.Hardtanh(lb,ub),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# warp image according to the grid
class waspWarper(nn.Module):
    def __init__(self, opt):
        super(waspWarper, self).__init__()
        self.opt = opt
        self.batchSize = opt.batchSize
        self.imgSize = opt.imgSize

    def forward(self, input_img, input_grid):
        self.warp = input_grid.permute(0,2,3,1)
        self.output = F.grid_sample(input_img, self.warp)
        return self.output

class waspPerspectiveProjector(nn.Module):
    def __init__(self, opt):
        super(waspPerspectiveProjector, self).__init__()
        self.opt = opt
        self.batchSize = opt.batchSize
        self.imgSize   = opt.imgSize
    def forward(self, pp_pars, ones_vector, w_basegrid, w_depth, ones_mat):
        # Get the overall grid. 
        overall_grid = torch.cat((w_basegrid, w_depth, ones_mat), 1).view(ones_vector.size(0), 4, -1)
        # Transform the pp_pars to 12-dim by adding a 1. 
        M_pars       = torch.cat((pp_pars, ones_vector), 1).view(-1, 3, 4)

        # Project. 
        proj_grid    = torch.bmm(M_pars, overall_grid)
        proj_grid_x  = proj_grid[:,0,:]/proj_grid[:,2,:]
        proj_grid_y  = proj_grid[:,1,:]/proj_grid[:,2,:]
        outputs      = torch.cat((proj_grid_x.unsqueeze(1), proj_grid_y.unsqueeze(1)), 1).contiguous()
        return outputs.view(ones_vector.size(0), 2, self.imgSize, self.imgSize)

class waspPerspectiveProjector2(nn.Module):
    def __init__(self, opt):
        super(waspPerspectiveProjector2, self).__init__()
        self.opt = opt
        self.batchSize = opt.batchSize
        self.imgSize   = opt.imgSize
    def forward(self, M_pars, w_basegrid, w_depth, ones_mat):
        # Get the overall grid. 
        overall_grid = torch.cat((w_basegrid, w_depth, ones_mat), 1).view(M_pars.size(0), 4, -1)
        # Transform the pp_pars to 12-dim by adding a 1. 
#        M_pars       = torch.cat((pp_pars, ones_vector), 1).view(-1, 3, 4)

        # Project. 
        proj_grid    = torch.bmm(M_pars, overall_grid)
        proj_grid_x  = proj_grid[:,0,:]/proj_grid[:,2,:]
        proj_grid_y  = proj_grid[:,1,:]/proj_grid[:,2,:]
        outputs      = torch.cat((proj_grid_x.unsqueeze(1), proj_grid_y.unsqueeze(1)), 1).contiguous()
        return outputs.view(M_pars.size(0), 2, self.imgSize, self.imgSize), proj_grid

class waspPerspectiveProjector3(nn.Module):
    def __init__(self, opt):
        super(waspPerspectiveProjector3, self).__init__()
        self.opt = opt
        self.batchSize = opt.batchSize
        self.imgSize   = opt.imgSize
    def forward(self, texture, M_pars, w_basegrid, w_depth, ones_mat):
        # Get the overall grid. 
        overall_grid = torch.cat((w_basegrid, w_depth, ones_mat), 1).view(M_pars.size(0), 4, -1)
        # Transform the pp_pars to 12-dim by adding a 1. 
#        M_pars       = torch.cat((pp_pars, ones_vector), 1).view(-1, 3, 4)

        # Project. 
        proj_grid    = torch.bmm(M_pars, overall_grid)
#        proj_grid_x  = proj_grid[:,0,:]/proj_grid[:,2,:]
#        proj_grid_y  = proj_grid[:,1,:]/proj_grid[:,2,:]
#        outputs      = torch.cat((proj_grid_x.unsqueeze(1), proj_grid_y.unsqueeze(1)), 1).contiguous()

        camera_coords = proj_grid[:,0:3,:].data.cpu().numpy()
        f_grid        = camera_coords.reshape([-1, 3, self.imgSize, self.imgSize])
        outputs       = np.zeros((M_pars.size(0), self.opt.nc, self.imgSize, self.imgSize))
        tex           = texture.data.cpu().numpy()
        for iid in range(M_pars.size(0)):
            outputs[iid,:,:,:] = my_inverse_warp(tex[iid,:,:,:], f_grid[iid,:,:], self.opt)
        outputs = torch.from_numpy(outputs).type('torch.FloatTensor').cuda()
        outputs = Variable(outputs, requires_grad=False)
        return outputs
#        return outputs.view(M_pars.size(0), 2, self.imgSize, self.imgSize), proj_grid

class waspPerspectiveProjector4(nn.Module):
    def __init__(self, opt):
        super(waspPerspectiveProjector4, self).__init__()
        self.opt = opt
        self.batchSize = opt.batchSize
        self.imgSize   = opt.imgSize
    def forward(self, M_pars, mesh_points, ovec):
        # Get the overall grid. 
        overall_grid = mesh_points.view(M_pars.size(0), 4, -1)

        # Get the projection matrix from these parameters
        proj_matrix = get_projection_matrix(M_pars, ovec)

        # Project. 
        camera_coords   = torch.bmm(proj_matrix, overall_grid)
        # This is what we return. 
        return camera_coords

# Get projection matrix given perspective parameters. 
def get_projection_matrix(M_pars, ovec):
    # Create a zero vector, of the same size as the ones vector. 
    zvec = ovec.clone()
    zvec = 0*ovec
    # Extract and adjust translation parameters. 
    trans_x     = M_pars[:,0].unsqueeze(1)
    trans_y     = M_pars[:,1].unsqueeze(1)
    trans_z     = M_pars[:,2].unsqueeze(1)
    # Create rotation matrix. 
    # Compute sines and cosines first. 
    c_theta_x   = torch.cos(M_pars[:,3]).unsqueeze(1)
    s_theta_x   = torch.sin(M_pars[:,3]).unsqueeze(1)
    c_theta_y   = torch.cos(M_pars[:,4]).unsqueeze(1)
    s_theta_y   = torch.sin(M_pars[:,4]).unsqueeze(1)
    c_theta_z   = torch.cos(M_pars[:,5]).unsqueeze(1)
    s_theta_z   = torch.sin(M_pars[:,5]).unsqueeze(1)
    # Create rotation matrices for each axis. 
    rot_mat_x   = torch.cat((torch.cat((ovec, zvec, zvec),dim=1).unsqueeze(1),
                             torch.cat((zvec, c_theta_x, s_theta_x),dim=1).unsqueeze(1),
                             torch.cat((zvec, -1*s_theta_x, c_theta_x),dim=1).unsqueeze(1)), dim=1)
    rot_mat_y   = torch.cat((torch.cat((c_theta_y, zvec, -1*s_theta_y),dim=1).unsqueeze(1),
                             torch.cat((zvec, ovec, zvec),dim=1).unsqueeze(1),
                             torch.cat((s_theta_y, zvec, c_theta_y),dim=1).unsqueeze(1)), dim=1)
    rot_mat_z   = torch.cat((torch.cat((c_theta_z, s_theta_z, zvec),dim=1).unsqueeze(1),
                             torch.cat((-1*s_theta_z, c_theta_z, zvec),dim=1).unsqueeze(1),
                             torch.cat((zvec, zvec, ovec),dim=1).unsqueeze(1)), dim=1)
    # The rotation matrix now a product of these three. 
    rot_mat     = torch.bmm(torch.bmm(rot_mat_x, rot_mat_y), rot_mat_z)
    # Now we augment rot_mat with the translation parameters to make the projection matrix. 
    proj_matrix = torch.cat((rot_mat, 
                                 torch.cat((trans_x, trans_y, trans_z),dim=1).unsqueeze(1).permute(0,2,1)), dim=2)
        # Augment the projection matrix with another row. 
#        proj_matrix = torch.cat((proj_matrix,
#                                 torch.cat((zvec, zvec, zvec, ovec),dim=1).unsqueeze(1)), dim=1)
    return proj_matrix


# A class that renders a texture according to a set of projected points. 
class waspRenderer(nn.Module):
    def __init__(self, opt):
        super(waspRenderer, self).__init__()
        self.opt        = opt
        self.imgSize    = opt.imgSize
        # Get the list of faces which will be used by the renderer. 
        self.F          = get_tri_list(self.imgSize)
        self.F1         = self.F[:,0]
        self.F2         = self.F[:,1]
        self.F3         = self.F[:,2]
        # self.tcoords is are dummy texture coordinates. 
        self.tcoords    = np.ones((4096,2))
        # We need a matrix which will add an appropriate value to indices of vertex_*_img so 
        #    that they can be retrieved from flat_*
        self.base_ivec  = self.imgSize*self.imgSize*(torch.arange(0,opt.batchSize).type('torch.LongTensor'))
        self.base_ivec  = self.base_ivec.view(-1,1).expand(opt.batchSize,self.imgSize*self.imgSize)
        # Meshgrid to compute BCs.
        self.x, self.y  = np.meshgrid(np.arange(self.imgSize), np.arange(self.imgSize))
        self.x, self.y  = torch.FloatTensor(self.x.astype(np.float)).cuda(), torch.FloatTensor(self.y.astype(np.float)).cuda()
        self.x, self.y  = Variable(self.x, requires_grad=False), Variable(self.y, requires_grad=False)

    def forward(self, vertices, textures):
        # vertices and textures are Variables of shape
        #   (opt.batchSize, 4, opt.imgSize*opt.imgSize) and (opt.batchSize, opt.nc, opt.imgSize, opt.imgSize),
        #   respectively. 
        batch_size          = vertices.size(0)
        img_size            = textures.size(2)

        # Rearrange and rescale vertices. 
        vertices            = (1 + vertices[:,[1,0,2],:])*self.imgSize*0.5

        # Slice base_vec so that we only use as much as necessary. 
        base_vec            = self.base_ivec[:batch_size,:].cuda()

        # Flatten the R, G, and B components of the texture images. 
        flat_R              = textures[:,0,:,:].contiguous().view(-1).contiguous()#,self.imgSize*self.imgSize)
        flat_G              = textures[:,1,:,:].contiguous().view(-1).contiguous()#,self.imgSize*self.imgSize)
        flat_B              = textures[:,2,:,:].contiguous().view(-1).contiguous()#,self.imgSize*self.imgSize)

        # Flatten the vertices so that they are easily accessible for the computation
        #   of barycentric coordinates. 
        flat_Vx             = vertices[:,0,:].contiguous().view(-1).contiguous()
        flat_Vy             = vertices[:,1,:].contiguous().view(-1).contiguous()
        
        # Create a matrix to hold the barycentric coordinates.
        barycentric_coords  = np.zeros((batch_size, 3, self.imgSize, self.imgSize))
        # Mask images for contributing pixels. 
        mask_images         = np.zeros((batch_size, 1, self.imgSize, self.imgSize))
        # Create arrays to hold vertex IDs. 
        vertex_1_img        = np.zeros((batch_size, self.imgSize*self.imgSize), dtype=np.int)
        vertex_2_img        = np.zeros((batch_size, self.imgSize*self.imgSize), dtype=np.int)
        vertex_3_img        = np.zeros((batch_size, self.imgSize*self.imgSize), dtype=np.int)
        # Copy vertices and textures onto the CPU, so that they can be handled by menpo. 
        vertices_cpu        = vertices.data.cpu().numpy().transpose(0,2,1)
        textures_cpu        = textures.data.cpu().numpy()

        # Adjust the vertices_cpu to conform to menpo standards. 
#        vertices_cpu        = (1 + vertices_cpu[:,:,[1,0,2]])*self.imgSize*0.5
        # Get the barycentric coordinates for every image in the batch. 
        for b in range(batch_size):
            mesh            = TexturedTriMesh(vertices_cpu[b,:,:],
                                              self.tcoords, 
                                              mimage.Image(textures_cpu[b,:,:,:]), 
                                              trilist=self.F)
            ret             = rasterize_barycentric_coordinate_images(mesh, [self.imgSize, self.imgSize])

            # Save barycentric coordinates and face IDs. 
            barycentric_coords[b,:,:,:] = ret[0].pixels
            face_ids                    = ret[1].pixels.squeeze().flatten()
            mask_images[b,:,:,:]        = ret[0].mask.pixels
            # Save vertex ID images. 
            vertex_1_img[b,:]           = self.F1[face_ids]
            vertex_2_img[b,:]           = self.F2[face_ids]
            vertex_3_img[b,:]           = self.F3[face_ids]
 
        # Convert vertex_*_img into torch.LongTensors. 
        vertex_1_img    = torch.LongTensor(vertex_1_img).cuda()
        vertex_2_img    = torch.LongTensor(vertex_2_img).cuda()
        vertex_3_img    = torch.LongTensor(vertex_3_img).cuda()

        # Convert the mask image. 
        mask_images     = Variable(torch.FloatTensor(mask_images).cuda(), requires_grad=False)

        # Coordinates in the flattened vectors. 
        flat_coords_1   = (base_vec + vertex_1_img).view(-1)
        flat_coords_2   = (base_vec + vertex_2_img).view(-1)
        flat_coords_3   = (base_vec + vertex_3_img).view(-1)

        # Copy the barycentric coordinates into a Variable
        v_bcoords   = Variable(torch.FloatTensor(barycentric_coords).cuda(), requires_grad=False)

        # Retrieve the X and Y coordinates of the projected points according to the chosen faces. 
        X_1         = flat_Vx[flat_coords_1].view(batch_size, self.imgSize, self.imgSize)
        X_2         = flat_Vx[flat_coords_2].view(batch_size, self.imgSize, self.imgSize)
        X_3         = flat_Vx[flat_coords_3].view(batch_size, self.imgSize, self.imgSize)
        Y_1         = flat_Vy[flat_coords_1].view(batch_size, self.imgSize, self.imgSize)
        Y_2         = flat_Vy[flat_coords_2].view(batch_size, self.imgSize, self.imgSize)
        Y_3         = flat_Vy[flat_coords_3].view(batch_size, self.imgSize, self.imgSize)

        # Compute the denominator
        Mex         = self.y.expand(batch_size, self.imgSize, self.imgSize)
        Nex         = self.x.expand(batch_size, self.imgSize, self.imgSize)
        h           = (Y_2 - Y_3)*(X_1 - X_3) + (X_3 - X_2)*(Y_1 - Y_3)
        lambda_1    = ((Y_2 - Y_3)*(Mex - X_3) + (X_3 - X_2)*(Nex - Y_3))/h
        lambda_2    = ((Y_3 - Y_1)*(Mex - X_3) + (X_1 - X_3)*(Nex - Y_3))/h
        lambda_3    = 1.0 - lambda_1 - lambda_2
    
        # Rearrange and restack. 
        lambda_1    = lambda_1.unsqueeze(1)
        lambda_1[mask_images != 1] = 0
        lambda_2    = lambda_2.unsqueeze(1)
        lambda_2[mask_images != 1] = 0
        lambda_3    = lambda_3.unsqueeze(1)
        lambda_3[mask_images != 1] = 0
        m_bcoords   = torch.cat((lambda_1, lambda_2, lambda_3), dim=1)
       
        # Verify whether they are correct. 
        m = torch.mean(torch.abs(m_bcoords - v_bcoords)).data[0]
        print('Error in barycentric coordinate computation: %10f' %(m))

        # Slice the Variables so that we can directly use vertex_*_img without having
        #    to iterate over all images in the batch. 
        red_v1      = flat_R[flat_coords_1].view(batch_size, self.imgSize, self.imgSize).contiguous()
        red_v2      = flat_R[flat_coords_2].view(batch_size, self.imgSize, self.imgSize).contiguous()
        red_v3      = flat_R[flat_coords_3].view(batch_size, self.imgSize, self.imgSize).contiguous()

        green_v1    = flat_G[flat_coords_1].view(batch_size, self.imgSize, self.imgSize).contiguous()
        green_v2    = flat_G[flat_coords_2].view(batch_size, self.imgSize, self.imgSize).contiguous()
        green_v3    = flat_G[flat_coords_3].view(batch_size, self.imgSize, self.imgSize).contiguous()

        blue_v1     = flat_B[flat_coords_1].view(batch_size, self.imgSize, self.imgSize).contiguous()
        blue_v2     = flat_B[flat_coords_2].view(batch_size, self.imgSize, self.imgSize).contiguous()
        blue_v3     = flat_B[flat_coords_3].view(batch_size, self.imgSize, self.imgSize).contiguous()

        # Now interpolate using the barycentric coordinates. 
        red_final   = m_bcoords[:,0,:,:]*red_v1   + m_bcoords[:,1,:,:]*red_v2   + m_bcoords[:,2,:,:]*red_v3
        green_final = m_bcoords[:,0,:,:]*green_v1 + m_bcoords[:,1,:,:]*green_v2 + m_bcoords[:,2,:,:]*green_v3
        blue_final  = m_bcoords[:,0,:,:]*blue_v1  + m_bcoords[:,1,:,:]*blue_v2  + m_bcoords[:,2,:,:]*blue_v3

        # outputs are concatenation of these three images. 
        outputs     = torch.cat((red_final.unsqueeze(1), green_final.unsqueeze(1), blue_final.unsqueeze(1)),dim=1)
        return outputs 

    def backward_dummy(self, dLdoutput):
        
        pass

# A class that renders a texture according to a set of projected points. 
# User-defined gradients. 
class customRenderer_F(Function):
    @staticmethod
    def forward(ctx, vertices, textures, opt):
        # vertices and textures are Variables of shape
        #   (opt.batchSize, 4, opt.imgSize*opt.imgSize) and (opt.batchSize, opt.nc, opt.imgSize, opt.imgSize),
        #   respectively. 

        # Batch size of the input. 
        batch_size          = vertices.size(0)
        img_size            = textures.size(2)

        imgSize     = img_size
        # Get the list of faces which will be used by the renderer. 
        F           = get_tri_list(img_size)
        F1          = F[:,0]
        F2          = F[:,1]
        F3          = F[:,2]
        # self.tcoords is are dummy texture coordinates. 
        tcoords     = np.ones((img_size*img_size,2))
        # We need a matrix which will add an appropriate value to indices of vertex_*_img so 
        #    that they can be retrieved from flat_*
        
        # 
        base_ivec   = img_size*img_size*(torch.arange(0,batch_size).type('torch.LongTensor'))
        base_ivec   = base_ivec.view(-1,1).expand([batch_size, img_size*img_size])
        # Slice base_vec so that we only use as much as necessary. 
        base_vec    = base_ivec.cuda()

        # Meshgrid to compute BCs.
        x, y        = np.meshgrid(np.arange(img_size), np.arange(img_size))
        x, y        = torch.FloatTensor(x.astype(np.float)).cuda(), torch.FloatTensor(y.astype(np.float)).cuda()

        # Rearrange and rescale vertices. 
        vertices            = (1 + vertices[:,[1,0,2],:])*img_size*0.5

        # Flatten the R, G, and B components of the texture images. 
        flat_R              = textures[:,0,:,:].contiguous().view(-1).contiguous()#,imgSize*imgSize)
        flat_G              = textures[:,1,:,:].contiguous().view(-1).contiguous()#,imgSize*imgSize)
        flat_B              = textures[:,2,:,:].contiguous().view(-1).contiguous()#,imgSize*imgSize)

        # Flatten the vertices so that they are easily accessible for the computation
        #   of barycentric coordinates. 
        flat_Vx             = vertices[:,0,:].contiguous().view(-1).contiguous()
        flat_Vy             = vertices[:,1,:].contiguous().view(-1).contiguous()
        
        # Create a matrix to hold the barycentric coordinates.
        barycentric_coords  = np.zeros((batch_size, 3, img_size, img_size))
        # Create a matrix to hold the denominators in barycentric coordinates' equations. 
        phi_values          = np.zeros((batch_size, 1, img_size, img_size))
        # Mask images for contributing pixels. 
        mask_images         = np.zeros((batch_size, 1, img_size, img_size))
        # Create arrays to hold vertex IDs. 
        vertex_1_img        = np.zeros((batch_size, img_size*img_size), dtype=np.int)
        vertex_2_img        = np.zeros((batch_size, img_size*img_size), dtype=np.int)
        vertex_3_img        = np.zeros((batch_size, img_size*img_size), dtype=np.int)
        # Copy vertices and textures onto the CPU, so that they can be handled by menpo. 
        vertices_cpu        = vertices.cpu().numpy().transpose(0,2,1)
        textures_cpu        = textures.cpu().numpy()

        # Adjust the vertices_cpu to conform to menpo standards. 
#        vertices_cpu        = (1 + vertices_cpu[:,:,[1,0,2]])*imgSize*0.5
        # Get the barycentric coordinates for every image in the batch. 
        for b in range(batch_size):
            gc.collect()
            mesh            = TexturedTriMesh(vertices_cpu[b,:,:],
                                              tcoords, 
                                              mimage.Image(textures_cpu[b,:,:,:]), 
                                              trilist=F)
            ret             = my_rasterize_barycentric_coordinate_images(mesh, [img_size, img_size])

            # Save barycentric coordinates and face IDs. 
            barycentric_coords[b,:,:,:] = ret[0].pixels
            face_ids                    = ret[1].pixels.squeeze().flatten()
            mask_images[b,0,:,:]        = ret[0].mask.pixels
            phi_values[b,0,:,:]         = ret[2]
            # Save vertex ID images. 
            vertex_1_img[b,:]           = F1[face_ids]
            vertex_2_img[b,:]           = F2[face_ids]
            vertex_3_img[b,:]           = F3[face_ids]
 
        # Convert vertex_*_img into torch.LongTensors. 
        vertex_1_img    = torch.LongTensor(vertex_1_img).cuda()
        vertex_2_img    = torch.LongTensor(vertex_2_img).cuda()
        vertex_3_img    = torch.LongTensor(vertex_3_img).cuda()

        # Convert to variables. 
        mask_images     = torch.FloatTensor(mask_images).cuda()
        phi_values      = torch.FloatTensor(phi_values).cuda()
#        phi_values      = Variable(torch.FloatTensor(phi_values).cuda(). requires_grad=False)

        # Coordinates in the flattened vectors. 
        flat_coords_1   = (base_vec + vertex_1_img).view(-1)
        flat_coords_2   = (base_vec + vertex_2_img).view(-1)
        flat_coords_3   = (base_vec + vertex_3_img).view(-1)

        # Copy the barycentric coordinates into a Variable
        v_bcoords   = torch.FloatTensor(barycentric_coords).cuda()

        # Retrieve the X and Y coordinates of the projected points according to the chosen faces. 
        X_1         = flat_Vx[flat_coords_1].view(batch_size, imgSize, imgSize)
        X_2         = flat_Vx[flat_coords_2].view(batch_size, imgSize, imgSize)
        X_3         = flat_Vx[flat_coords_3].view(batch_size, imgSize, imgSize)
        Y_1         = flat_Vy[flat_coords_1].view(batch_size, imgSize, imgSize)
        Y_2         = flat_Vy[flat_coords_2].view(batch_size, imgSize, imgSize)
        Y_3         = flat_Vy[flat_coords_3].view(batch_size, imgSize, imgSize)

        # Verify whether they are correct. 
#        m = torch.mean(torch.abs(m_bcoords - v_bcoords)).data[0]
#        print('Error in barycentric coordinate computation: %10f' %(m))

        # Slice the Variables so that we can directly use vertex_*_img without having
        #    to iterate over all images in the batch. 
        red_v1      = flat_R[flat_coords_1].view(batch_size, imgSize, imgSize).contiguous()
        red_v2      = flat_R[flat_coords_2].view(batch_size, imgSize, imgSize).contiguous()
        red_v3      = flat_R[flat_coords_3].view(batch_size, imgSize, imgSize).contiguous()

        green_v1    = flat_G[flat_coords_1].view(batch_size, imgSize, imgSize).contiguous()
        green_v2    = flat_G[flat_coords_2].view(batch_size, imgSize, imgSize).contiguous()
        green_v3    = flat_G[flat_coords_3].view(batch_size, imgSize, imgSize).contiguous()

        blue_v1     = flat_B[flat_coords_1].view(batch_size, imgSize, imgSize).contiguous()
        blue_v2     = flat_B[flat_coords_2].view(batch_size, imgSize, imgSize).contiguous()
        blue_v3     = flat_B[flat_coords_3].view(batch_size, imgSize, imgSize).contiguous()

        # Now interpolate using the barycentric coordinates. 
        red_final   = v_bcoords[:,0,:,:]*red_v1   + v_bcoords[:,1,:,:]*red_v2   + v_bcoords[:,2,:,:]*red_v3
        green_final = v_bcoords[:,0,:,:]*green_v1 + v_bcoords[:,1,:,:]*green_v2 + v_bcoords[:,2,:,:]*green_v3
        blue_final  = v_bcoords[:,0,:,:]*blue_v1  + v_bcoords[:,1,:,:]*blue_v2  + v_bcoords[:,2,:,:]*blue_v3

        # outputs are concatenation of these three images. 
        outputs     = torch.cat((red_final.unsqueeze(1), green_final.unsqueeze(1), blue_final.unsqueeze(1)),dim=1)

        ctx.base_vec        = base_vec
        ctx.vertex_1_img    = vertex_1_img
        ctx.vertex_2_img    = vertex_2_img
        ctx.vertex_3_img    = vertex_3_img
        ctx.v_bcoords       = v_bcoords
        ctx.mask_images     = mask_images
        ctx.phi_values      = phi_values
        ctx.X_1             = X_1
        ctx.X_2             = X_2
        ctx.X_3             = X_3
        ctx.Y_1             = Y_1
        ctx.Y_2             = Y_2
        ctx.Y_3             = Y_3

        ctx.save_for_backward(textures)
        # Save some variables for gradient computation. 
#        ctx.save_for_backward(base_vec, flat_coords_1, flat_coords_2, flat_coords_3, v_bcoords, \
#                red_v1, red_v2, red_v3, green_v1, green_v2, green_v3, blue_v1, blue_v2, \
#                blue_v3, mask_images, phi_values, X_1, X_2, X_3, Y_1, Y_2, Y_3, F)
        return outputs 

    @staticmethod
    def forward_torch(ctx, vertices, textures, opt):
        # vertices and textures are Variables of shape
        #   (opt.batchSize, 4, opt.imgSize*opt.imgSize) and (opt.batchSize, opt.nc, opt.imgSize, opt.imgSize),
        #   respectively. 

        # Batch size of the input. 
        batch_size          = vertices.size(0)
        img_size            = textures.size(2)

        imgSize     = img_size
        # Get the list of faces which will be used by the renderer. 
        F           = get_tri_list(img_size)
        F1          = F[:,0]
        F2          = F[:,1]
        F3          = F[:,2]
        # self.tcoords is are dummy texture coordinates. 
        tcoords     = np.ones((img_size*img_size,2))
        # We need a matrix which will add an appropriate value to indices of vertex_*_img so 
        #    that they can be retrieved from flat_*
        
        # 
        base_ivec   = img_size*img_size*(torch.arange(0,batch_size).type('torch.LongTensor'))
        base_ivec   = base_ivec.view(-1,1).expand(batch_size, img_size*img_size)
        # Slice base_vec so that we only use as much as necessary. 
        base_vec            = base_ivec.cuda()

        # Meshgrid to compute BCs.
        x, y        = np.meshgrid(np.arange(img_size), np.arange(img_size))
        x, y        = torch.FloatTensor(x.astype(np.float)).cuda(), torch.FloatTensor(y.astype(np.float)).cuda()

        # Rearrange and rescale vertices. 
        vertices            = (1 + vertices[:,[1,0,2],:])*img_size*0.5

        # Flatten the R, G, and B components of the texture images. 
        flat_R              = textures[:,0,:,:].contiguous().view(-1).contiguous()#,imgSize*imgSize)
        flat_G              = textures[:,1,:,:].contiguous().view(-1).contiguous()#,imgSize*imgSize)
        flat_B              = textures[:,2,:,:].contiguous().view(-1).contiguous()#,imgSize*imgSize)

        # Flatten the vertices so that they are easily accessible for the computation
        #   of barycentric coordinates. 
        flat_Vx             = vertices[:,0,:].contiguous().view(-1).contiguous()
        flat_Vy             = vertices[:,1,:].contiguous().view(-1).contiguous()
        
        # Create a matrix to hold the barycentric coordinates.
        barycentric_coords  = np.zeros((batch_size, 3, img_size, img_size))
        # Create a matrix to hold the denominators in barycentric coordinates' equations. 
        phi_values          = np.zeros((batch_size, 1, img_size, img_size))
        # Mask images for contributing pixels. 
        mask_images         = np.zeros((batch_size, 1, img_size, img_size))
        # Create arrays to hold vertex IDs. 
        vertex_1_img        = np.zeros((batch_size, img_size*img_size), dtype=np.int)
        vertex_2_img        = np.zeros((batch_size, img_size*img_size), dtype=np.int)
        vertex_3_img        = np.zeros((batch_size, img_size*img_size), dtype=np.int)
        # Copy vertices and textures onto the CPU, so that they can be handled by menpo. 
        vertices_cpu        = vertices.cpu().numpy().transpose(0,2,1)
        textures_cpu        = textures.cpu().numpy()

        # Adjust the vertices_cpu to conform to menpo standards. 
#        vertices_cpu        = (1 + vertices_cpu[:,:,[1,0,2]])*imgSize*0.5
        # Get the barycentric coordinates for every image in the batch. 
        for b in range(batch_size):
            mesh            = TexturedTriMesh(vertices_cpu[b,:,:],
                                              tcoords, 
                                              mimage.Image(textures_cpu[b,:,:,:]), 
                                              trilist=F)
            ret             = my_rasterize_barycentric_coordinate_images(mesh, [img_size, img_size])

            # Save barycentric coordinates and face IDs. 
            barycentric_coords[b,:,:,:] = ret[0].pixels
            face_ids                    = ret[1].pixels.squeeze().flatten()
            mask_images[b,0,:,:]        = ret[0].mask.pixels
            phi_values[b,0,:,:]         = ret[2]
            # Save vertex ID images. 
            vertex_1_img[b,:]           = F1[face_ids]
            vertex_2_img[b,:]           = F2[face_ids]
            vertex_3_img[b,:]           = F3[face_ids]
 
        # Convert vertex_*_img into torch.LongTensors. 
        vertex_1_img    = torch.LongTensor(vertex_1_img).cuda()
        vertex_2_img    = torch.LongTensor(vertex_2_img).cuda()
        vertex_3_img    = torch.LongTensor(vertex_3_img).cuda()

        # Convert to variables. 
        mask_images     = torch.FloatTensor(mask_images).cuda()
        phi_values      = torch.FloatTensor(phi_values).cuda()
#        phi_values      = Variable(torch.FloatTensor(phi_values).cuda(). requires_grad=False)

        # Coordinates in the flattened vectors. 
        flat_coords_1   = (base_vec + vertex_1_img).view(-1)
        flat_coords_2   = (base_vec + vertex_2_img).view(-1)
        flat_coords_3   = (base_vec + vertex_3_img).view(-1)

        # Copy the barycentric coordinates into a Variable
        v_bcoords   = torch.FloatTensor(barycentric_coords).cuda()

        # Retrieve the X and Y coordinates of the projected points according to the chosen faces. 
        X_1         = flat_Vx[flat_coords_1].view(batch_size, imgSize, imgSize)
        X_2         = flat_Vx[flat_coords_2].view(batch_size, imgSize, imgSize)
        X_3         = flat_Vx[flat_coords_3].view(batch_size, imgSize, imgSize)
        Y_1         = flat_Vy[flat_coords_1].view(batch_size, imgSize, imgSize)
        Y_2         = flat_Vy[flat_coords_2].view(batch_size, imgSize, imgSize)
        Y_3         = flat_Vy[flat_coords_3].view(batch_size, imgSize, imgSize)

        # Compute the denominator
        Mex         = y.expand(batch_size, imgSize, imgSize)
        Nex         = x.expand(batch_size, imgSize, imgSize)
        h           = (Y_2 - Y_3)*(X_1 - X_3) + (X_3 - X_2)*(Y_1 - Y_3)
        lambda_1    = ((Y_2 - Y_3)*(Mex - X_3) + (X_3 - X_2)*(Nex - Y_3))/h
        lambda_2    = ((Y_3 - Y_1)*(Mex - X_3) + (X_1 - X_3)*(Nex - Y_3))/h
        lambda_3    = 1.0 - lambda_1 - lambda_2
    
        # Rearrange and restack. 
        lambda_1    = (lambda_1.unsqueeze(1))*mask_images
        lambda_2    = (lambda_2.unsqueeze(1))*mask_images
        lambda_3    = (lambda_3.unsqueeze(1))*mask_images
        m_bcoords   = torch.cat((lambda_1, lambda_2, lambda_3), dim=1)
       
        # Verify whether they are correct. 
#        m = torch.mean(torch.abs(m_bcoords - v_bcoords)).data[0]
#        print('Error in barycentric coordinate computation: %10f' %(m))

        # Slice the Variables so that we can directly use vertex_*_img without having
        #    to iterate over all images in the batch. 
        red_v1      = flat_R[flat_coords_1].view(batch_size, imgSize, imgSize).contiguous()
        red_v2      = flat_R[flat_coords_2].view(batch_size, imgSize, imgSize).contiguous()
        red_v3      = flat_R[flat_coords_3].view(batch_size, imgSize, imgSize).contiguous()

        green_v1    = flat_G[flat_coords_1].view(batch_size, imgSize, imgSize).contiguous()
        green_v2    = flat_G[flat_coords_2].view(batch_size, imgSize, imgSize).contiguous()
        green_v3    = flat_G[flat_coords_3].view(batch_size, imgSize, imgSize).contiguous()

        blue_v1     = flat_B[flat_coords_1].view(batch_size, imgSize, imgSize).contiguous()
        blue_v2     = flat_B[flat_coords_2].view(batch_size, imgSize, imgSize).contiguous()
        blue_v3     = flat_B[flat_coords_3].view(batch_size, imgSize, imgSize).contiguous()

        # Now interpolate using the barycentric coordinates. 
        red_final   = v_bcoords[:,0,:,:]*red_v1   + v_bcoords[:,1,:,:]*red_v2   + v_bcoords[:,2,:,:]*red_v3
        green_final = v_bcoords[:,0,:,:]*green_v1 + v_bcoords[:,1,:,:]*green_v2 + v_bcoords[:,2,:,:]*green_v3
        blue_final  = v_bcoords[:,0,:,:]*blue_v1  + v_bcoords[:,1,:,:]*blue_v2  + v_bcoords[:,2,:,:]*blue_v3

        # outputs are concatenation of these three images. 
        outputs     = torch.cat((red_final.unsqueeze(1), green_final.unsqueeze(1), blue_final.unsqueeze(1)),dim=1)

        ctx.base_vec        = base_vec
        ctx.flat_coords_1   = flat_coords_1
        ctx.flat_coords_2   = flat_coords_2
        ctx.flat_coords_3   = flat_coords_3
        ctx.v_bcoords       = v_bcoords
        ctx.mask_images     = mask_images
        ctx.phi_values      = phi_values
        ctx.X_1             = X_1
        ctx.X_2             = X_2
        ctx.X_3             = X_3
        ctx.Y_1             = Y_1
        ctx.Y_2             = Y_2
        ctx.Y_3             = Y_3

        ctx.save_for_backward(textures)
        # Save some variables for gradient computation. 
#        ctx.save_for_backward(base_vec, flat_coords_1, flat_coords_2, flat_coords_3, v_bcoords, \
#                red_v1, red_v2, red_v3, green_v1, green_v2, green_v3, blue_v1, blue_v2, \
#                blue_v3, mask_images, phi_values, X_1, X_2, X_3, Y_1, Y_2, Y_3, F)
        return outputs 


    @staticmethod
    def backward(ctx, dLdoutput):
        # This version uses Numpy. Calculations are easier. 
        # dLdoutput has shape (batch_size, n_channels, img_size, img_size)
        batch_size, n_channels, img_size, _ = dLdoutput.size()

        # Retrieve saved variables. 
        base_vec        = ctx.base_vec.view(-1).cpu().numpy()
        vertex_1_img    = ctx.vertex_1_img.cpu().numpy()
        vertex_2_img    = ctx.vertex_2_img.cpu().numpy()
        vertex_3_img    = ctx.vertex_3_img.cpu().numpy()
        v_bcoords       = ctx.v_bcoords.cpu().numpy()
        mask_images     = ctx.mask_images.cpu().numpy()
        phi_values      = ctx.phi_values.cpu().numpy()
        X_1             = ctx.X_1.cpu().numpy()
        X_2             = ctx.X_2.cpu().numpy()
        X_3             = ctx.X_3.cpu().numpy()
        Y_1             = ctx.Y_1.cpu().numpy()
        Y_2             = ctx.Y_2.cpu().numpy()
        Y_3             = ctx.Y_3.cpu().numpy()

        textures        = ctx.saved_variables[0].view(-1, 3, img_size*img_size).data.cpu().numpy()
        dLdoutput       = dLdoutput.view(-1, 3, img_size*img_size).data.cpu().numpy()

#        base_vec, flat_coords_1, flat_coords_2, flat_coords_3, v_bcoords, red_v1, red_v2, red_v3, green_v1, \
#            green_v2, green_v3, blue_v1, blue_v2, blue_v3, mask_images, phi_values, outputs, X_1, \
#            X_2, X_3, Y_1, Y_2, Y_3, F = ctx.saved_variables
        dLdtexture  = np.zeros((batch_size, n_channels, img_size*img_size))
        dLdvertices = np.zeros((batch_size, 3, img_size*img_size))

        vertex_1_img = vertex_1_img.reshape([batch_size, img_size, img_size])
        vertex_2_img = vertex_2_img.reshape([batch_size, img_size, img_size])
        vertex_3_img = vertex_3_img.reshape([batch_size, img_size, img_size])
    
        lambda_1    = v_bcoords[:,0,:,:]#.view(-1, img_size*img_size)
        lambda_2    = v_bcoords[:,1,:,:]#.view(-1, img_size*img_size)
        lambda_3    = v_bcoords[:,2,:,:]#.view(-1, img_size*img_size)


        for x in range(img_size):
            for y in range(img_size):
                v_bts = np.where(mask_images[:,0,x,y] == 1)[0]
                if np.sum(v_bts) == 0:
                    # No images in this batch have anything at (x,y)
                    continue

                v_id    = x*img_size + y
                x_1     = X_1[v_bts,x,y]
                x_2     = X_2[v_bts,x,y]
                x_3     = X_3[v_bts,x,y]
                y_1     = Y_1[v_bts,x,y]
                y_2     = Y_2[v_bts,x,y]
                y_3     = Y_3[v_bts,x,y]

                l_1     = lambda_1[v_bts,x,y]
                l_2     = lambda_2[v_bts,x,y]
                l_3     = lambda_3[v_bts,x,y]

                i_1     = vertex_1_img[v_bts,x,y]
                i_2     = vertex_2_img[v_bts,x,y]
                i_3     = vertex_3_img[v_bts,x,y]

                h       = phi_values[v_bts,0,x,y]

                tdiff_1_3 = np.mean((textures[v_bts,:,i_1] - textures[v_bts,:,i_3])*dLdoutput[v_bts,:,v_id], axis=1)
                tdiff_2_3 = np.mean((textures[v_bts,:,i_2] - textures[v_bts,:,i_3])*dLdoutput[v_bts,:,v_id], axis=1)
#                tdiff_1_3       = torch.mean((textures[:,:,x_1,y_1] - textures[:,:,x_3,y_3])*dLdoutput[:,:,x,y],dim=1)[v_bts]     # might use sum here. 
#                tdiff_2_3       = torch.mean((textures[:,:,x_2,y_2] - textures[:,:,x_3,y_3])*dLdoutput[:,:,x,y],dim=1)[v_bts]

                # Fill derivatives w.r.t. points. 
                dLdvertices[v_bts,0,i_1] += -(l_1*(y_2-y_3)/h)*tdiff_1_3 + \
                                                ((y_3-y)/h - l_2/h*(y_2-y_3))*tdiff_2_3
                dLdvertices[v_bts,1,i_1] += -(l_1*(x_3-x_2)/h)*tdiff_1_3 + \
                                                ((x_3-x)/h - l_2/h*(x_3-x_2))*tdiff_2_3
                    
                dLdvertices[v_bts,0,i_2] += ((y_3-y)/h - l_1*(y_3-y_1)/h)*tdiff_1_3 - \
                                                (l_2*(y_3-y_1)/h)*tdiff_2_3
                dLdvertices[v_bts,1,i_2] += ((x-x_3)/h - l_1*(x_1-x_3)/h)*tdiff_1_3 - \
                                                (l_2*(x_1-x_3)/h)*tdiff_2_3

                dLdvertices[v_bts,0,i_3] += ((y-y_2)/h - l_1*(y_1-y_2)/h)*tdiff_1_3 + \
                                                ((y_1-y)/h - l_2*(y_1-y_2)/h)*tdiff_2_3
                dLdvertices[v_bts,1,i_3] += ((x_2-x)/h - l_1*(x_2-x_1)/h)*tdiff_1_3 + \
                                                    ((x-x_1)/h - l_2*(x_2-x_1)/h)*tdiff_2_3

                # Fill derivatives w.r.t. texture. 
                dLdtexture[v_bts,0,i_1] += l_1; dLdtexture[v_bts,1,i_1] += l_1; dLdtexture[v_bts,2,i_1] += l_1
                dLdtexture[v_bts,0,i_2] += l_2; dLdtexture[v_bts,1,i_2] += l_2; dLdtexture[v_bts,2,i_2] += l_2
                dLdtexture[v_bts,0,i_3] += l_3; dLdtexture[v_bts,1,i_3] += l_3; dLdtexture[v_bts,2,i_3] += l_3
       
        # Finally, multiply dLdtexture by dLdoutput
        dLdtexture = dLdoutput*dLdtexture
        dLdtexture = dLdtexture.reshape([-1, 3, img_size, img_size])

        dLdvertices = torch.FloatTensor(dLdvertices).cuda()
        dLdtexture  = torch.FloatTensor(dLdtexture).cuda()

        return Variable(dLdvertices), Variable(dLdtexture), None

    @staticmethod
    def backward_torch(ctx, dLdoutput):
        # dLdoutput has shape (batch_size, n_channels, img_size, img_size)
        batch_size, n_channels, img_size, _ = dLdoutput.size()

        # Retrieve saved variables. 
        base_vec        = ctx.base_vec
        flat_coords_1   = ctx.flat_coords_1
        flat_coords_2   = ctx.flat_coords_2
        flat_coords_3   = ctx.flat_coords_3
        v_bcoords       = ctx.v_bcoords
        mask_images     = ctx.mask_images
        phi_values      = ctx.phi_values
        X_1             = ctx.X_1
        X_2             = ctx.X_2
        X_3             = ctx.X_3
        Y_1             = ctx.Y_1
        Y_2             = ctx.Y_2
        Y_3             = ctx.Y_3

        textures        = ctx.saved_variables[0].view(-1, 3, img_size*img_size).data
        dLdoutput       = dLdoutput.view(-1, 3, img_size*img_size).data

#        base_vec, flat_coords_1, flat_coords_2, flat_coords_3, v_bcoords, red_v1, red_v2, red_v3, green_v1, \
#            green_v2, green_v3, blue_v1, blue_v2, blue_v3, mask_images, phi_values, outputs, X_1, \
#            X_2, X_3, Y_1, Y_2, Y_3, F = ctx.saved_variables
        dLdtexture  = torch.FloatTensor(batch_size, n_channels, img_size*img_size).fill_(0).cuda()
        dLdvertices = torch.FloatTensor(batch_size, 3, img_size*img_size).fill_(0).cuda()

        vertex_1_img = (flat_coords_1 - base_vec).view(batch_size, img_size, img_size)
        vertex_2_img = (flat_coords_2 - base_vec).view(batch_size, img_size, img_size)
        vertex_3_img = (flat_coords_3 - base_vec).view(batch_size, img_size, img_size)
    
        lambda_1    = v_bcoords[:,0,:,:]#.view(-1, img_size*img_size)
        lambda_2    = v_bcoords[:,1,:,:]#.view(-1, img_size*img_size)
        lambda_3    = v_bcoords[:,2,:,:]#.view(-1, img_size*img_size)


        for x in range(img_size):
            for y in range(img_size):
                v_bts = (mask_images[:,0,x,y] == 1)
                if torch.sum(v_bts) == 0:
                    # No images in this batch have anything at (x,y)
                    continue

                v_id    = x*img_size + y
                x_1     = X_1[:,x,y]#[v_bts]
                x_2     = X_2[:,x,y]#[v_bts]
                x_3     = X_3[:,x,y]#[v_bts]
                y_1     = Y_1[:,x,y]#[v_bts]
                y_2     = Y_2[:,x,y]#[v_bts]
                y_3     = Y_3[:,x,y]#[v_bts]

                l_1     = lambda_1[:,x,y]#[v_bts]
                l_2     = lambda_2[:,x,y]#[v_bts]
                l_3     = lambda_3[:,x,y]#[v_bts]

                i_1     = vertex_1_img[:,x,y]#[v_bts]
                i_2     = vertex_2_img[:,x,y]#[v_bts]
                i_3     = vertex_3_img[:,x,y]#[v_bts]

                h       = phi_values[:,0,x,y]#[v_bts]

#                tdiff_1_3       = torch.FloatTensor(v_bts.size(0)).fill_(0).cuda()
#                tdiff_2_3       = torch.FloatTensor(v_bts.size(0)).fill_(0).cuda()
                for i, b in enumerate(v_bts):
                    if not b:
                        # Nothing to do here. Move along, folks. 
                        continue    

                    tdiff_1_3 = torch.mean((textures[i,:,i_1[i]] - textures[i,:,i_3[i]])*dLdoutput[i,:,v_id])
                    tdiff_2_3 = torch.mean((textures[i,:,i_2[i]] - textures[i,:,i_3[i]])*dLdoutput[i,:,v_id])
#                tdiff_1_3       = torch.mean((textures[:,:,x_1,y_1] - textures[:,:,x_3,y_3])*dLdoutput[:,:,x,y],dim=1)[v_bts]     # might use sum here. 
#                tdiff_2_3       = torch.mean((textures[:,:,x_2,y_2] - textures[:,:,x_3,y_3])*dLdoutput[:,:,x,y],dim=1)[v_bts]

                    # Fill derivatives w.r.t. points. 
                    dLdvertices[i,0,i_1[i]] += -(l_1[i]*(y_2[i]-y_3[i])/h[i])*tdiff_1_3 + \
                                                    ((y_3[i]-y)/h[i] - l_2[i]/h[i]*(y_2[i]-y_3[i]))*tdiff_2_3
                    dLdvertices[i,1,i_1[i]] += -(l_1[i]*(x_3[i]-x_2[i])/h[i])*tdiff_1_3 + \
                                                    ((x_3[i]-x)/h[i] - l_2[i]/h[i]*(x_3[i]-x_2[i]))*tdiff_2_3
                    
                    dLdvertices[i,0,i_2[i]] += ((y_3[i]-y)/h[i] - l_1[i]*(y_3[i]-y_1[i])/h[i])*tdiff_1_3 - \
                                                    (l_2[i]*(y_3[i]-y_1[i])/h[i])*tdiff_2_3
                    dLdvertices[i,1,i_2[i]] += ((x-x_3[i])/h[i] - l_1[i]*(x_1[i]-x_3[i])/h[i])*tdiff_1_3 - \
                                                    (l_2[i]*(x_1[i]-x_3[i])/h[i])*tdiff_2_3

                    dLdvertices[i,0,i_3[i]] += ((y-y_2[i])/h[i] - l_1[i]*(y_1[i]-y_2[i])/h[i])*tdiff_1_3 + \
                                                    ((y_1[i]-y)/h[i] - l_2[i]*(y_1[i]-y_2[i])/h[i])*tdiff_2_3
                    dLdvertices[i,1,i_3[i]] += ((x_2[i]-x)/h[i] - l_1[i]*(x_2[i]-x_1[i])/h[i])*tdiff_1_3 + \
                                                    ((x-x_1[i])/h[i] - l_2[i]*(x_2[i]-x_1[i])/h[i])*tdiff_2_3

                    # Fill derivatives w.r.t. texture. 
                    dLdtexture[i,0,i_1[i]] += l_1[i]; dLdtexture[i,1,i_1[i]] += l_1[i]; dLdtexture[i,2,i_1[i]] += l_1[i]
                    dLdtexture[i,1,i_2[i]] += l_2[i]; dLdtexture[i,1,i_2[i]] += l_2[i]; dLdtexture[i,2,i_2[i]] += l_2[i]
                    dLdtexture[i,0,i_3[i]] += l_3[i]; dLdtexture[i,1,i_3[i]] += l_3[i]; dLdtexture[i,2,i_3[i]] += l_3[i]
       
        # Finally, multiply dLdtexture by dLdoutput
        dLdtexture = dLdoutput*dLdtexture
        dLdtexture = dLdtexture.view(-1, 3, img_size, img_size)

        return Variable(dLdvertices), Variable(dLdtexture), None


#            dLdvertices[b,0,:]

class customRenderer(nn.Module):
    def __init__(self, opt):
        super(customRenderer, self).__init__()
        self.opt = opt

    def forward(self, vertices, textures):
        return customRenderer_F.apply(vertices, textures, self.opt)


# waspAffineGrid: Generate an affine grid from affine transform parameters. 
class waspAffineGrid(nn.Module):
    def __init__(self, opt):
        super(waspAffineGrid, self).__init__()
        self.batchSize = opt.batchSize
        self.imgSize   = opt.imgSize

    def forward(self, af_pars, basegrid):
#        output_grid = F.affine_grid(af_pars, torch.Size((self.batchSize, 3, self.imgSize, self.imgSize))).permute(0,3,1,2)
#        return output_grid
#def getAffineGrid(af_pars, basegrid):
        nBatch, nc, iS, _ = basegrid.size()
        affine = af_pars.expand(iS, iS, nBatch, 6).permute(2,3,0,1).contiguous()
        afft_x = affine[:,0,:,:]*basegrid[:,0,:,:] + affine[:,1,:,:]*basegrid[:,1,:,:] + affine[:,2,:,:]
        afft_y = affine[:,3,:,:]*basegrid[:,0,:,:] + affine[:,4,:,:]*basegrid[:,1,:,:] + affine[:,5,:,:]
        afft_x = afft_x.unsqueeze(1)
        afft_y = afft_y.unsqueeze(1)
        output_grid = torch.cat((afft_x, afft_y), 1)
        return output_grid

# convolutional joint net for corr and decorr

class waspConvJoint0(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

# convolutional joint net for corr and decorr

class waspConvResblockJoint0(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvResblockJoint0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 


# convolutional joint net for corr and decorr

class waspConvJoint9(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint9, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        #output = self.mixer(output)
        return output 


class waspConvJointConditional0(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJointConditional0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.opt = opt
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz+self.opt.injdim,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2, inputc):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output0 = self.main(input0).view(-1,self.nz)
        output1 = torch.cat(output0, inputc,1)
        output = self.mixer(output1)
        return output 

# convolutional joint net for corr and decorr

class waspFadeJoint0(nn.Module):
    def __init__(self, opt,  nz1 =128, nz2 = 128, nz = 256):
        super(waspFadeJoint0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.nz1 = nz1
        self.nz2 = nz2
        self.main = nn.Sequential(
            nn.Linear(self.nz1 + self.nz2, self.nz),
            nn.ReLU()
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0)
        output = self.mixer(output)
        return output 


# convolutional joint net for corr and decorr

class waspConvJointLsq0(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJointLsq0, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1)
            #nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

# convolutional joint net for corr and decorr

class waspConvJoint(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

class waspConvJoint2(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint2, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.ReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 

class waspConvJoint3(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint3, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 


class waspConvJoint4(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32, nz = 256):
        super(waspConvJoint4, self).__init__()
        self.ngpu = opt.ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.nz, 4, 4, 0, bias=False)
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.nz,1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        input0 = torch.cat((((input1+1)/2), input2), 1)
        output = self.main(input0).view(-1,self.nz)
        output = self.mixer(output)
        return output 
# convolutional joint net for corr and decorr

class waspConvJointTanh(nn.Module):
    def __init__(self, opt,  nc=2, ndf = 32):
        super(waspConvJointTanh, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Tanh(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Tanh(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.Tanh(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 128, 4, 4, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input1, input2):
        input0 = torch.cat((input1, input2), 1)
        output = self.main(input0)
        return output   


# The encoders
class Encoders(nn.Module):
    def __init__(self, opt):
        super(Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp

# The encoders
class EncodersSlicer(nn.Module):
    def __init__(self, opt):
        super(EncodersSlicer, self).__init__()
        self.opt= opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = self.opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim, pend = self.opt.idim + opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp

# Encoder that uses encodes only the affine transformation. 
class EncodersAffine(nn.Module):
    def __init__(self, opt):
        super(EncodersAffine, self).__init__()
        self.opt  = opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart=0, pend=self.opt.idim)
        self.zAmixer = waspSlicer(opt, ngpu=1, pstart=opt.idim, pend=opt.zdim)
    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zAffT = self.zAmixer(self.z)
        return self.z, self.zImg, self.zAffT


# Encoder that also encodes the affine transform.
class EncodersResAffine(nn.Module):
    def __init__(self, opt):
        super(EncodersResAffine, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart=0, pend=self.opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart=self.opt.idim, pend=self.opt.idim+self.opt.wdim)
        self.zAmixer = waspSlicer(opt, ngpu=1, pstart=self.opt.idim+self.opt.wdim, pend=self.opt.zdim)
    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        self.zAffT = self.zAmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zAffT


# Encoder that also encodes the affine transform.
class EncodersAffineIntegral(nn.Module):
    def __init__(self, opt):
        super(EncodersAffineIntegral, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspSlicer(opt, ngpu=1, pstart=0, pend=self.opt.idim)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart=self.opt.idim, pend=self.opt.idim+self.opt.wdim)
        self.zAmixer = waspSlicer(opt, ngpu=1, pstart=self.opt.idim+self.opt.wdim, pend=self.opt.zdim)
    def forward(self, input):
        self.z     = self.encoder(input)
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        self.zAffT = self.zAmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zAffT

class EncodersAffineIntegralDepth(nn.Module):
    def __init__(self, opt):
        super(EncodersAffineIntegralDepth, self).__init__()
        self.opt     = opt
        self.ngpu    = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.idim)       # 2D texture latent space. 
        self.zDmixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.ddim)       # 2D depth latent space. 
        self.zWmixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.wdim)       # 2D texture warping latent space. 
        self.zXmixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.xdim)       # 2D depth warping latent space. 
        self.zPmixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.pdim)       # 11-dim perspective params latent space. 
    def forward(self, inputs):
        self.z       = self.encoder(inputs)
        self.zImg    = self.zImixer(self.z)
        self.zDepth  = self.zDmixer(self.z)
        self.zWarp   = self.zWmixer(self.z)
        self.zDWarp  = self.zXmixer(self.z)
        self.zPersPr = self.zPmixer(self.z)
        return self.z, self.zImg, self.zDepth, self.zWarp, self.zDWarp, self.zPersPr

class EncodersIntegralBCWarper(nn.Module):
    def __init__(self, opt):
        super(EncodersIntegralBCWarper, self).__init__()
        self.opt     = opt
        self.ngpu    = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.idim)       # 2D texture latent space. 
        self.zWmixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.wdim)       # 2D texture warping latent space. 
    def forward(self, inputs):
        self.z       = self.encoder(inputs)
        self.zImg    = self.zImixer(self.z)
        self.zWarp   = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp


class EncodersPerspectiveBCWarperCamera(nn.Module):
    def __init__(self, opt):
        super(EncodersPerspectiveBCWarperCamera, self).__init__()
        self.opt     = opt
        self.ngpu    = opt.ngpu
        self.encoder = waspEncoder(opt, ngpu=1, nc=opt.nc, ndf=opt.ndf, ndim=opt.zdim)
        self.zImixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.idim)       # 2D texture latent space. 
        self.zDmixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.ddim)
        self.zPmixer = waspNonLinearity(opt, ngpu=1, nin=opt.zdim, nout=opt.pdim)       # Camera params. 
    def forward(self, inputs):
        self.z       = self.encoder(inputs)
        self.zImg    = self.zImixer(self.z)
        self.zDepth  = self.zDmixer(self.z)
        self.zPers   = self.zPmixer(self.z)
        return self.z, self.zImg, self.zDepth, self.zPers


# The encoders
class EncodersInject(nn.Module):
    def __init__(self, opt):
        super(EncodersInject, self).__init__()
        self.opt=opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderInject(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf, ndim = opt.zdim, injdim = opt.injdim)
        self.zImixer = waspMixer(opt, ngpu=1, nin = opt.injdim+opt.zdim, nout = opt.idim)
        self.zWmixer = waspMixer(opt, ngpu=1, nin = opt.injdim+opt.zdim, nout = opt.wdim)

    def forward(self, input):
        self.z     = self.encoder(input)
        self.zLabel = self.z[:,0:self.opt.injdim]
        self.zImg  = self.zImixer(self.z)
        self.zWarp = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zLabel

# The encoders
class EncodersInject2(nn.Module):
    def __init__(self, opt):
        super(EncodersInject2, self).__init__()
        self.opt=opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderInject2(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf)
        self.zLmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.zdim_inj, pend = (self.opt.zdim_inj+self.opt.injdim))
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = (self.opt.zdim_inj+self.opt.injdim))
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.zdim_inj, pend = (self.opt.zdim_inj+self.opt.zdim_inj+self.opt.injdim))
    def forward(self, input):
        self.z      = self.encoder(input)
        self.zLabel = self.zLmixer(self.z)
        self.zImg   = self.zImixer(self.z)
        self.zWarp  = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zLabel

# The encoders
class EncodersInject3(nn.Module):
    def __init__(self, opt):
        super(EncodersInject3, self).__init__()
        self.opt=opt
        self.ngpu = opt.ngpu
        self.encoder = waspEncoderInject2(opt, ngpu=1, nc=opt.nc, ndf = opt.ndf)
        self.zLmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim, pend = (self.opt.idim_inj))
        self.zImixer = waspSlicer(opt, ngpu=1, pstart = 0, pend = self.opt.idim_inj)
        self.zWmixer = waspSlicer(opt, ngpu=1, pstart = self.opt.idim, pend = opt.idim+opt.wdim+opt.injdim)
    def forward(self, input):
        self.z      = self.encoder(input)
        self.zLabel = self.zLmixer(self.z)
        self.zImg   = self.zImixer(self.z)
        self.zWarp  = self.zWmixer(self.z)
        return self.z, self.zImg, self.zWarp, self.zLabel


# The decoders
#class Decoders(nn.Module):
#    def __init__(self, opt):
#        super(Decoders, self).__init__()
#        self.ngpu     = opt.ngpu
#        self.idim = opt.idim
#        self.wdim = opt.wdim
#        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub = 1)
#        self.decoderW = waspDecoder(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=-1, ub=1)
#        self.warper   = waspWarper(opt)
#    def forward(self, zI, zW):
#        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
#        self.warping = self.decoderW(zW.view(-1,self.wdim,1,1))
#        self.output  = self.warper(self.texture, self.warping)
#        return self.texture, self.warping, self.output

# The decoders that use residule warper
class DecodersResWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersResWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.resWarping = self.decoderW(zW.view(-1,self.wdim,1,1))
        self.resWarping = self.resWarping*2-1
        self.warping = self.resWarping + basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output

# The decoders that use residule warper
class DecodersResWarperInject2(nn.Module):
    def __init__(self, opt):
        super(DecodersResWarperInject2, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim_inj, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim_inj, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.opt.idim_inj,1,1))
        self.resWarping = self.decoderW(zW.view(-1,self.opt.wdim_inj,1,1))
        self.resWarping = self.resWarping*2-1
        self.warping = self.resWarping + basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output


class L2Reg(nn.Module):
    def __init__(self, opt):
        super(L2Reg, self).__init__()
        self.opt = opt
        self.criterion = nn.MSELoss()
    def forward(self, x, zero, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        if zero is not None:
            self.loss = w*self.criterion(x, zero)
        else:
            self.loss = w*torch.mean(x*x)
        return self.loss

class PPL2Reg(nn.Module):
    def __init__(self, opt):
        super(PPL2Reg, self).__init__()
        self.opt = opt
        self.criterion = nn.MSELoss()
    def forward(self, x, zero, w1=1, w2=1):
        w1 = torch.cuda.FloatTensor(1).fill_(w1)
        w2 = torch.cuda.FloatTensor(1).fill_(w2)
        if self.opt.cuda:
            w1.cuda()
            w2.cuda()
        w1 = Variable(w1, requires_grad=False)
        w2 = Variable(w2, requires_grad=False)
        if zero is not None:
            self.loss = w1*self.criterion(x[:,:3], zero[:,:3]) + \
                        w2*self.criterion(x[:,3:], zero[:,3:])
        else:
            self.loss = w1*torch.mean(x[:,:3]*x[:,:3]) + w2*torch.mean(x[:,3:]*x[:,3:])
        return self.loss



class TotalVaryLoss(nn.Module):
    def __init__(self,opt):
        super(TotalVaryLoss, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + 
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return self.loss

class SelfSmoothLoss(nn.Module):
    def __init__(self,opt):
        super(SelfSmoothLoss, self).__init__()
        self.opt = opt
    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.y_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.loss = torch.norm(self.x_diff) + torch.norm(self.y_diff)
        self.loss = w * self.loss
        return self.loss        

class WeightMSELoss(nn.Module):
    def __init__(self,opt):
        super(WeightMSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.opt = opt
    def forward(self, input, target, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w*self.criterion(input, target)
        return self.loss


class WeightABSLoss(nn.Module):
    def __init__(self,opt):
        super(WeightABSLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.opt=opt
    def forward(self, input, target, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w*self.criterion(input, target)
        return self.loss

class WeightBCELoss(nn.Module):
    def __init__(self,opt):
        super(WeightBCELoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.opt=opt
    def forward(self, input, target, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w*self.criterion(input, target)
        return self.loss

class BiasReduceLoss(nn.Module):
    def __init__(self,opt):
        super(BiasReduceLoss, self).__init__()
        self.opt = opt
        self.criterion = nn.MSELoss()
    def forward(self, x, y, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.avg = torch.sum(x,0).unsqueeze(0)
        self.loss = w*self.criterion(self.avg, y)
        return self.loss


class WaspGridSpatialIntegral0(nn.Module):
    def __init__(self,opt):
        super(WaspGridSpatialIntegral0, self).__init__()
        self.opt = opt
        self.w = self.opt.imgSize
        self.filterx = torch.cuda.FloatTensor(1,1,self.w,self.w).fill_(0)
        self.filtery = torch.cuda.FloatTensor(1,1,self.w,self.w).fill_(0)
        self.filterx[:,:,-1,:] = 1
        self.filtery[:,:,:,-1] = 1
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if opt.cuda:
            self.filterx.cuda()
            self.filtery.cuda()
    def forward(self, input_diffgrid):
        #print(input_diffgrid.size())
        fullx = F.conv2d(input_diffgrid[:,0,:,:].unsqueeze(1), self.filterx, stride=1, padding=self.w-1)
        fully = F.conv2d(input_diffgrid[:,1,:,:].unsqueeze(1), self.filtery, stride=1, padding=self.w-1)
        output_grid = torch.cat((fullx[:,:,0:self.w,0:self.w], fully[:,:,0:self.w,0:self.w]),1)
        return output_grid


class WaspGridSpatialIntegral(nn.Module):
    def __init__(self,opt):
        super(WaspGridSpatialIntegral, self).__init__()
        self.opt = opt
        self.w = self.opt.imgSize
        self.filterx = torch.cuda.FloatTensor(1,1,1,self.w).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1,1,self.w,1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if opt.cuda:
            self.filterx.cuda()
            self.filtery.cuda()
    def forward(self, input_diffgrid):
        #print(input_diffgrid.size())
        fullx = F.conv_transpose2d(input_diffgrid[:,0,:,:].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:,1,:,:].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:,:,0:self.w,0:self.w], fully[:,:,0:self.w,0:self.w]),1)
        return output_grid

# The decoders that use residule warper
#class DecodersResWarper(nn.Module):
#    def __init__(self, opt):
#        super(DecodersResWarper, self).__init__()
#        self.ngpu = opt.ngpu
#        self.idim = opt.idim
#        self.wdim = opt.wdim
#        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
#        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
#        self.warper   = waspWarper(opt)
#    def forward(self, zI, zW, basegrid):
#        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
#        self.resWarping = self.decoderW(zW.view(-1,self.wdim,1,1))
#        self.resWarping = self.resWarping*2-1
#        self.warping = self.resWarping + basegrid
#        self.output  = self.warper(self.texture, self.warping)
#        return self.texture, self.resWarping, self.output

# The decoders that use residule warper
class DecodersSleepyWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersSleepyWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.ReLUx = nn.ReLU()
        self.ReLUy = nn.ReLU()
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.resWarping = self.decoderW(zW.view(-1,self.wdim,1,1))
        self.resWarping = self.resWarping*2-1
        self.warping = self.resWarping + basegrid

        self.warping_grad_x = self.warping[:, 0, :, 1:] - self.warping[:, 0, :, :-1] 
        self.warping_grad_y = self.warping[:, 1, 1:, :] - self.warping[:, 1, :-1, :]
        self.warping_grad_x_pos = self.ReLUx(self.warping_grad_x) 
        self.warping_grad_y_pos = self.ReLUy(self.warping_grad_y) 

        self.warping[:, 0, :, 1:] = self.warping[:, 0, :, :-1] + self.warping_grad_x_pos
        self.warping[:, 1, 1:, :] = self.warping[:, 1, :-1, :] + self.warping_grad_y_pos

        self.resWarping_pos = self.warping - basegrid
        
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping_pos, self.output


# Decoders that use only the affine transformation. 
class DecodersAffineWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersAffineWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderA = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.affineGrid = waspAffineGrid(opt)
        self.warper   = waspWarper(opt)
    def forward(self, zI, zA, basegrid):
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.warping = self.affineGrid(self.af_pars, basegrid)
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.warping, self.output, self.af_pars

# The decoders that use residule warper
class DecodersIntegralWarperInj(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarperInj, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.opt  = opt
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim+opt.injdim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim+opt.injdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim+self.opt.injdim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim+self.opt.injdim,1,1))*0.1
        self.warping = self.integrator(self.diffentialWarping)-2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping



# The decoders that use residule warper
class DecodersIntegralWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08          # was *0.1
        self.warping = self.integrator(self.diffentialWarping)-1.2                      # was - 2 
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping

# The decoders that use residule warper
class DecodersIntegralWarper0(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper0, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoderSigm(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping


# The decoders that use residule warper
class DecodersIntegralWarper1(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper1, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping

# The decoders that use residule warper
class DecodersIntegralWarper2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping

# Decoders that use integral warper, as well as an affine transformation above. 
class DecodersAffineIntegralWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersAffineIntegralWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.decoderA = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode affine params, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Transform the texture. 
        self.af_tex  = self.warper(self.texture, self.affine)
        # Decode and integrate the warping grid. 
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        # Apply the warping grid to the deformed texture. 
        self.output  = self.warper(self.af_tex, self.warping)
        # Apply the warping grid to the transformed grid to get the final deformation field.
        self.warp_af = self.warper(self.affine, self.warping)
        # Get the residual deformation field.
        self.resWarping = self.warp_af - self.affine #self.warping - basegrid
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars

        # Decoders that use residual warper, as well as an affine transformation above. 
class DecodersResAffineWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersResAffineWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.decoderA = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
    def forward(self, zI, zW, zA, basegrid):
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.resWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))
        self.resWarping = self.resWarping*2 - 1
        self.warping    = self.resWarping + basegrid
        self.wp_tex  = self.warper(self.texture, self.warping)
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        self.output  = self.warper(self.wp_tex, self.affine)
        self.warp_af = self.warper(self.warping, self.affine)
        self.resWarping = self.warp_af - self.affine
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars

# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersIntegralAffineWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.decoderA = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1, 1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        # Apply face deformation to texture. 
        self.wp_tex  = self.warper(self.texture, self.warping)
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper(self.wp_tex, self.affine)
        # Apply affine transformation to face warping to get the final deformation field.
        self.warp_af = self.warper(self.warping, self.affine)
        # Get the residual deformation.
        self.resWarping = self.warping - basegrid
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars

# Decoders that use affine and integral warping, as well as perspective projection. 
class DecodersIntegralAffineDepth(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineDepth, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.pdim = opt.pdim
        self.ddim = opt.ddim
        self.xdim = opt.xdim
        self.decoderI   = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW   = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.decoderD   = waspDepthDecoderTanh(opt, ngpu=self.ngpu, nz=opt.ddim, nc=1, ngf=opt.ngf, lb=-1, ub=1)
        self.decoderDW  = waspDecoderTanhNS(opt, ngpu=self.ngpu, nz=opt.xdim, nc=1, ngf=opt.ngf, lb=-0.1, ub=0.1)
        self.decoderP   = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.pdim, ndim=12)
        self.warper     = waspWarper(opt)
        self.persProj   = waspPerspectiveProjector3(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter     = nn.Hardtanh(-1, 1)

    def forward(self, zI, zD, zW, zX, zP, basegrid, ones_vector, ones_mat, zero_p):    
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping) - 1.2
        self.warping = self.cutter(self.warping)
        self.wp_tex  = self.warper(self.texture, self.warping)
        w_basegrid   = self.warper(basegrid, self.warping)
        self.m_depth = self.decoderD(zD.view(-1, self.ddim, 1, 1))      # Design choice - whether to use the same latent space for Z.
        self.m_wp    = self.decoderDW(zX.view(-1, self.xdim, 1, 1))
# -- OLD, which doesn't use griddata or tri.LinearTriInterpolator
#        self.pp_pars = self.decoderP(zP.view(-1, self.pdim))
#        self.f3dwarp = self.persProj(self.pp_pars+zero_p.expand_as(self.pp_pars), ones_vector, basegrid, self.m_depth + self.m_wp, ones_mat)
#        self.f3dwarp = self.warper(basegrid, self.f3dwarp)
#        self.output  = self.warper(self.wp_tex, self.f3dwarp)
# ---
        self.pp_pars = self.decoderP(zP.view(-1, self.pdim)).view(-1,3,4)
        self.output  = self.persProj(self.texture, self.pp_pars, basegrid, self.m_depth+self.m_wp, ones_mat)
        self.res_warping = self.warping - basegrid
        return self.texture, self.res_warping, self.output, self.m_depth, self.m_wp, self.pp_pars

# Decoders that use affine and integral warping, as well as perspective projection and waspRenderer
class DecodersIntegralAffineDepthRenderer(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineDepthRenderer, self).__init__()
        self.ngpu       = opt.ngpu
        self.idim       = opt.idim
        self.wdim       = opt.wdim
        self.pdim       = opt.pdim
        self.ddim       = opt.ddim
        self.xdim       = opt.xdim
        self.imgSize    = opt.imgSize

        self.decoderI   = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW   = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
#        self.decoderD   = waspDepthDecoderTanh(opt, ngpu=self.ngpu, nz=opt.ddim, nc=1, ngf=opt.ngf, lb=-1, ub=1)
        self.decoderD   = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.ddim, ndim=4096)
        self.decoderDW  = waspDecoderTanhNS(opt, ngpu=self.ngpu, nz=opt.xdim, nc=1, ngf=opt.ngf, lb=-1, ub=1)
        self.decoderP   = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.pdim, ndim=6)
        self.warper     = waspWarper(opt)
        self.persProj   = waspPerspectiveProjector4(opt)
        self.renderer   = waspRenderer(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter     = nn.Hardtanh(-1, 1)
        self.depth_z    = 1.0/self.ddim*Variable(torch.ones(1, self.ddim).type('torch.FloatTensor').cuda(), requires_grad=False)
#        self.depth_map  = Variable(torch.randn(1, 1, opt.imgSize, opt.imgSize).type('torch.FloatTensor').cuda(), requires_grad=True)

    def forward(self, zI, zD, zW, zX, zP, basegrid, ones_vector, ones_mat, zero_p):    
        self.texture    = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping    = self.integrator(self.differentialWarping) - 1.2
        self.warping    = self.cutter(self.warping)
#        self.wp_tex     = self.warper(self.texture, self.warping)      # Uncomment to include integral warping.    
        self.wp_tex     = self.texture                                # Uncomment to have no integral warping.
        w_basegrid      = self.warper(basegrid, self.warping)
#        self.m_depth    = self.decoderD(zD.view(-1, self.ddim, 1, 1))      # Design choice - whether to use the same latent space for Z.
#        self.m_depth    = self.depth_map.expand(basegrid.size(0), 1, self.imgSize, self.imgSize)
        self.m_depth    = self.decoderD(self.depth_z).view(1, 1, self.imgSize, self.imgSize).repeat(basegrid.size(0),1,1,1)
        self.m_wp       = self.decoderDW(zX.view(-1, self.xdim, 1, 1))
        self.pp_pars    = self.decoderP(zP.view(-1, self.pdim))
#        mesh_points     = torch.cat((basegrid, self.m_depth, ones_mat),dim=1)       # no +self.m_wp because there is no deformation. 
#        self.ccoords    = self.persProj(self.pp_pars, mesh_points, ones_vector)     # Renderer test. 
#        self.output     = self.renderer(self.ccoords, self.wp_tex)      # 
        self.warp_tri_test_coords = torch.cat((w_basegrid, 1*ones_mat), dim=1).view(-1, 3, self.imgSize*self.imgSize)
        self.output     = self.renderer(self.warp_tri_test_coords, self.wp_tex)      # 
#        self.output     = self.warper(self.texture, self.warping)
        self.res_warping = self.warping - basegrid
        return self.texture, self.wp_tex, self.res_warping, self.output, self.m_depth, self.m_wp, self.pp_pars

# Decoders that use affine and integral warping, as well as perspective projection and waspRenderer
class DecodersPerspectiveDepthRenderer2(nn.Module):
    def __init__(self, opt):
        super(DecodersPerspectiveDepthRenderer2, self).__init__()
        self.ngpu       = opt.ngpu
        self.idim       = opt.idim
        self.wdim       = opt.wdim
        self.pdim       = opt.pdim
        self.ddim       = opt.ddim
        self.xdim       = opt.xdim
        self.imgSize    = opt.imgSize

        self.decoderI   = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderD   = waspDepthDecoderTanh(opt, ngpu=self.ngpu, nz=opt.ddim, nc=1, ngf=opt.ngf, lb=-1, ub=1)
        self.decoderP   = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.pdim, ndim=6)
        self.persProj   = waspPerspectiveProjector4(opt)
        self.renderer   = waspRenderer(opt)

    def forward(self, zI, zD, zP, basegrid, ones_mat, ones_vector, zero_p):    
        self.texture    = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.m_depth    = (self.decoderD(zD.view(-1, self.ddim, 1, 1))-0.5)*0.5  # Design choice - whether to use the same latent space for Z.
        self.pp_pars    = self.decoderP(zP.view(-1, self.pdim))
        mesh_points     = torch.cat((basegrid, self.m_depth, ones_mat),dim=1)     # no +self.m_wp because there is no deformation. 
        self.ccoords    = self.persProj(self.pp_pars, mesh_points, ones_vector)   # Renderer test. 
        self.output     = self.renderer(self.ccoords, self.texture)
        return self.texture, self.output, self.m_depth, self.pp_pars


# Decoders that use affine and integral warping, as well as perspective projection and waspRenderer
class DecodersIntegralAffineDepthCustomRenderer(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineDepthCustomRenderer, self).__init__()
        self.ngpu       = opt.ngpu
        self.idim       = opt.idim
        self.wdim       = opt.wdim
        self.pdim       = opt.pdim
        self.ddim       = opt.ddim
        self.xdim       = opt.xdim
        self.imgSize    = opt.imgSize

        self.decoderI   = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW   = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
#        self.decoderD   = waspDepthDecoderTanh(opt, ngpu=self.ngpu, nz=opt.ddim, nc=1, ngf=opt.ngf, lb=-1, ub=1)
        self.decoderD   = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.ddim, ndim=4096)
        self.decoderDW  = waspDecoderTanhNS(opt, ngpu=self.ngpu, nz=opt.xdim, nc=1, ngf=opt.ngf, lb=-1, ub=1)
        self.decoderP   = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.pdim, ndim=6)
        self.warper     = waspWarper(opt)
        self.persProj   = waspPerspectiveProjector4(opt)
        self.renderer   = customRenderer(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter     = nn.Hardtanh(-1, 1)
        self.depth_z    = 1.0/self.ddim*Variable(torch.ones(1, self.ddim).type('torch.FloatTensor').cuda(), requires_grad=False)
#        self.depth_map  = Variable(torch.randn(1, 1, opt.imgSize, opt.imgSize).type('torch.FloatTensor').cuda(), requires_grad=True)

    def forward(self, zI, zD, zW, zX, zP, basegrid, ones_vector, ones_mat, zero_p):    
        self.texture    = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping    = self.integrator(self.differentialWarping) - 1.2
        self.warping    = self.cutter(self.warping)
#        self.wp_tex     = self.warper(self.texture, self.warping)      # Uncomment to include integral warping.    
        self.wp_tex     = self.texture
        w_basegrid      = self.warper(basegrid, self.warping)
#        self.m_depth    = self.decoderD(zD.view(-1, self.ddim, 1, 1))      # Design choice - whether to use the same latent space for Z.
#        self.m_depth    = self.depth_map.expand(basegrid.size(0), 1, self.imgSize, self.imgSize)
        self.m_depth    = self.decoderD(self.depth_z).view(1, 1, self.imgSize, self.imgSize).repeat(basegrid.size(0),1,1,1)
        self.m_wp       = self.decoderDW(zX.view(-1, self.xdim, 1, 1))
        self.pp_pars    = self.decoderP(zP.view(-1, self.pdim))
        mesh_points     = torch.cat((basegrid, self.m_depth+self.m_wp, ones_mat),dim=1)
        self.ccoords    = self.persProj(self.pp_pars, mesh_points, ones_vector)
        self.output     = self.renderer(self.ccoords, self.wp_tex)
        self.res_warping = self.warping - basegrid
        return self.texture, self.wp_tex, self.res_warping, self.output, self.m_depth, self.m_wp, self.pp_pars

# Decoders that use affine and integral warping, as well as perspective projection and waspRenderer
class DecodersIntegralBCWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralBCWarper, self).__init__()
        self.ngpu       = opt.ngpu
        self.idim       = opt.idim
        self.wdim       = opt.wdim
        self.pdim       = opt.pdim
        self.ddim       = opt.ddim
        self.xdim       = opt.xdim
        self.imgSize    = opt.imgSize

        self.decoderI   = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW   = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper     = waspWarper(opt)
        self.renderer   = waspRenderer(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter     = nn.Hardtanh(-1, 1)
        self.depth_z    = 1.0/self.ddim*Variable(torch.ones(1, self.ddim).type('torch.FloatTensor').cuda(), requires_grad=False)

    def forward(self, zI, zW, basegrid, ones_mat, zero_p):    
        self.texture    = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping    = self.integrator(self.differentialWarping) - 1.2
        self.warping    = self.cutter(self.warping)
        self.wp_tex     = self.texture                                # Uncomment to have no integral warping.
        self.warp_tri_test_coords = torch.cat((self.warping, 1*ones_mat), dim=1).view(-1, 3, self.imgSize*self.imgSize)
        self.output     = self.renderer(self.warp_tri_test_coords, self.wp_tex)      # 
        self.res_warping = self.warping - basegrid
        return self.texture, self.res_warping, self.output, self.wp_tex

# Decoders that use affine and integral warping, as well as perspective projection and waspRenderer
class DecodersIntegralCustomBCWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralCustomBCWarper, self).__init__()
        self.ngpu       = opt.ngpu
        self.idim       = opt.idim
        self.wdim       = opt.wdim
        self.pdim       = opt.pdim
        self.ddim       = opt.ddim
        self.xdim       = opt.xdim
        self.imgSize    = opt.imgSize

        self.decoderI   = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW   = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper     = waspWarper(opt)
        self.renderer   = customRenderer(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter     = nn.Hardtanh(-1, 1)
        self.depth_z    = 1.0/self.ddim*Variable(torch.ones(1, self.ddim).type('torch.FloatTensor').cuda(), requires_grad=False)

    def forward(self, zI, zW, basegrid, ones_mat, zero_p):    
        self.texture    = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping    = self.integrator(self.differentialWarping) - 2
        self.warping    = self.cutter(self.warping)
        self.wp_tex     = self.texture                                # Uncomment to have no integral warping.
        self.warp_tri_test_coords = torch.cat((self.warping, 1*ones_mat), dim=1).view(-1, 3, self.imgSize*self.imgSize)
        self.output     = self.renderer(self.warp_tri_test_coords, self.wp_tex)      # 
        self.res_warping = self.warping - basegrid
        return self.texture, self.res_warping, self.output, self.wp_tex


# Decoders that use affine and integral warping, as well as perspective projection and waspRenderer
class DecodersIntegralBCWarperCamera(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralBCWarperCamera, self).__init__()
        self.ngpu       = opt.ngpu
        self.idim       = opt.idim
        self.wdim       = opt.wdim
        self.pdim       = opt.pdim
        self.ddim       = opt.ddim
        self.xdim       = opt.xdim
        self.imgSize    = opt.imgSize

        self.decoderI   = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderP   = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.pdim, ndim=6)
        self.persProj   = waspPerspectiveProjector4(opt)
        self.renderer   = waspRenderer(opt)

    def forward(self, zI, zP, basegrid, ones_mat, ones_vector, zero_p):    
        self.texture    = self.decoderI(zI.view(-1, self.idim, 1, 1))
        self.pp_pars    = self.decoderP(zP.view(-1, self.pdim))
        mesh_points     = torch.cat((basegrid, ones_mat, ones_mat),dim=1)
        self.ccoords    = self.persProj(self.pp_pars, mesh_points, ones_vector)
        self.output     = self.renderer(self.ccoords, self.texture)
        return self.texture, self.output, self.pp_pars



# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersIntegralAffineCodeWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineCodeWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.decoderICode = waspDecoderTextureCode(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.tdf1, ngf=opt.ngf, lb=0, ub=1)
        self.decoderI     = FinalCodeDecoder(opt, ngpu=self.ngpu, ngf=opt.tdf1, nc=opt.nc, lb=0, ub=1)
        self.decoderW     = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.decoderA     = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderICode(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
        self.warping = self.integrator(self.differentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        # Apply face deformation to texture. 
        self.wp_tex  = self.warper(self.texture, self.warping)
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.outputC = self.warper(self.wp_tex, self.affine)
        self.output  = self.decoderI(self.outputC)
        self.texture = self.decoderI(self.texture)
        # Apply affine transformation to face warping to get the final deformation field.
        self.warp_af = self.warper(self.warping, self.affine)
        # Get the residual deformation.
        self.resWarping = self.warping - basegrid
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars


# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersIntegralAffineWarperPreTrain(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineWarperPreTrain, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.trW  = opt.trW
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.decoderA = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        if self.trW:
            self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
            self.warping = self.integrator(self.differentialWarping)-1.2
            self.warping = self.cutter(self.warping)
            # Apply face deformation to texture. 
            self.wp_tex  = self.warper(self.texture, self.warping)
        else:
            self.wp_tex  = self.texture
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper(self.wp_tex, self.affine)
        # Check if we compute the non-linear deformation. 
        if self.trW:
            # Apply affine transformation to face warping to get the final deformation field.
            self.warp_af = self.warper(self.warping, self.affine)
            # Get the residual deformation.
            self.resWarping = self.warping - basegrid
        else:
            self.warp_af = self.affine
            # Residual warping is None if self.trW is False. 
            self.resWarping = None
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars


# Decoders that use residual warper, as well as an affine transformation above. 
class DecodersIntegralAffineWarperFixAffine(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralAffineWarperFixAffine, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.adim = opt.adim
        self.trW  = opt.trW
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=0.1)
        self.decoderA = waspDecoderLinear(opt, ngpu=self.ngpu, nz=opt.adim, ndim=6)
        self.warper   = waspWarper(opt)
        self.affineGrid = waspAffineGrid(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, zA, basegrid):
        # Decode the texture. 
        self.texture = self.decoderI(zI.view(-1, self.idim, 1, 1))
        # Decode and integrate the face deformation. 
        if self.trW:
            self.differentialWarping = self.decoderW(zW.view(-1, self.wdim, 1, 1))*0.08
            self.warping = self.integrator(self.differentialWarping)-1.2
            self.warping = self.cutter(self.warping)
            # Apply face deformation to texture. 
            self.wp_tex  = self.warper(self.texture, self.warping)
        else:
            self.wp_tex  = self.texture
        # Decode the affine transformation, and get the affine grid. 
        self.af_pars = self.decoderA(zA.view(-1, self.adim))
        self.affine  = self.affineGrid(self.af_pars, basegrid)
        # Apply affine transformation to deformed texture. 
        self.output  = self.warper(self.wp_tex, self.affine)
        # Check if we compute the non-linear deformation. 
        if self.trW:
            # Apply affine transformation to face warping to get the final deformation field.
            self.warp_af = self.warper(self.warping, self.affine)
            # Get the residual deformation.
            self.resWarping = self.warping - basegrid
        else:
            self.warp_af = self.affine
            # Residual warping is None if self.trW is False. 
            self.resWarping = None
        return self.texture, self.resWarping, self.output, self.warp_af, self.af_pars


# The decoders that use residule warper
class DecodersIntegralWarperInject2(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarperInject2, self).__init__()
        self.opt = opt
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim_inj, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim_inj, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.opt.idim_inj,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.opt.wdim_inj,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping


# The decoders that use residule warper
class DecodersIntegralWarper2_B(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper2_B, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder_B(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh_B(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.2
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping

# The decoders that use residule warper
class DecodersIntegralWarper3(nn.Module):
    def __init__(self, opt):
        super(DecodersIntegralWarper3, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.integrator = WaspGridSpatialIntegral(opt)
        self.cutter = nn.Hardtanh(-1,1)
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.diffentialWarping = self.decoderW(zW.view(-1,self.wdim,1,1))*0.08
        self.warping = self.integrator(self.diffentialWarping)-1.01
        self.warping = self.cutter(self.warping)
        self.resWarping = self.warping-basegrid
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.warping


class DecodersSlopeWarper(nn.Module):
    def __init__(self, opt):
        super(DecodersSlopeWarper, self).__init__()
        self.ngpu = opt.ngpu
        self.idim = opt.idim
        self.wdim = opt.wdim
        self.decoderI = waspDecoder(opt, ngpu=self.ngpu, nz=opt.idim, nc=opt.nc, ngf=opt.ngf, lb=0, ub=1)
        self.decoderW = waspDecoderTanh(opt, ngpu=self.ngpu, nz=opt.wdim, nc=2, ngf=opt.ngf, lb=0, ub=1)
        self.warper   = waspWarper(opt)
        self.negaSlope = nn.ReLU()
    def forward(self, zI, zW, basegrid):
        self.texture = self.decoderI(zI.view(-1,self.idim,1,1))
        self.resWarping = self.decoderW(zW.view(-1,self.wdim,1,1))
        self.resWarping = self.resWarping*2-1
        self.warping = self.resWarping + basegrid

        self.warping_grad_x = self.warping[:, 0, :, :-1] - self.warping[:, 0, :, 1:] 
        self.warping_grad_y = self.warping[:, 1, :-1, :] - self.warping[:, 1, 1:, :]
        #print(self.warping_grad_x.size())
        #print(self.warping_grad_y.size())
        self.warping_grad = torch.cat((self.warping_grad_x.unsqueeze(1), self.warping_grad_y.permute(0,2,1).unsqueeze(1)),1).contiguous()
        self.slope = self.negaSlope(self.warping_grad)
        self.output  = self.warper(self.texture, self.warping)
        return self.texture, self.resWarping, self.output, self.slope


