from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import math
# our data loader
import WaspDataLoader
import gc
import vanillaLog
import numpy as np
#from logger import Logger
# my functions
#import zx

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=160, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default = True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--modelPath', default='', help="path to model (to continue training)")
parser.add_argument('--dirCheckpoints', default='.', help='folder to model checkpoints')
parser.add_argument('--dirImageoutput', default='.', help='folder to output images')
parser.add_argument('--dirTestingoutput', default='.', help='folder to testing results/images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--epoch_iter', type=int,default=200, help='number of epochs on entire dataset')
parser.add_argument('--location', type = int, default=0, help ='where is the code running')
parser.add_argument('-f',type=str,default= '', help='dummy input required for jupyter notebook')
opt = parser.parse_args()

opt.output_dir_prefix = './results/'
opt.data_dir_prefix = '../data/3DMM_syn_2000/'

# misc
opt.modelPath = ''#'results/checkpoints/CelebAnoHisteq_persProj_baseline_z64/wasp_model_epoch_92'
opt.dirCheckpoints   =   opt.output_dir_prefix + 'checkpoints/3DMMsyn_persProj_baseline_z32_renderer2'
opt.dirImageoutput   =   opt.output_dir_prefix + 'images/3DMMsyn_persProj_baseline_z32_renderer2'
opt.dirTestingoutput =   opt.output_dir_prefix + 'testing/3DMMsyn_persProj_baseline_z32_renderer2'
opt.dirLogger        =   './tblogs/3DMMsyn_persProj_baseline_z32_renderer2'
opt.vanillaLogger    =   './vlogs/3DMMsyn_persProj_baseline_z32_renderer2'
opt.vanillaLogFile   = opt.vanillaLogger + '/errors.txt'

opt.imgSize=64
opt.cuda = True
opt.use_dropout = 0
opt.ngf = 16
opt.ndf = 16
opt.idim = 1
opt.wdim = 32
opt.ddim = 1
opt.xdim = 32
opt.pdim = 32
opt.zdim = opt.idim + opt.ddim + opt.pdim
opt.use_gpu = True
opt.gpu_ids = 3
opt.ngpu = 1
opt.nc = 3
opt.dynamicWeight = 2
opt.dynamicWeight_tau = 0.01
opt.fix_texture = False
print(opt.gpu_ids)
print(opt)

try:
    os.makedirs(opt.dirCheckpoints)
except OSError:
    pass
try:
    os.makedirs(opt.dirImageoutput)
except OSError:
    pass
try:
    os.makedirs(opt.dirTestingoutput)
except OSError:
    pass
try:
    os.makedirs(opt.dirLogger)
except OSError:
    pass
try:
    os.makedirs(opt.vanillaLogger)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


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

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# sampe iamges
def visualizeAsImages(img_list, output_dir, 
                      n_sample=4, id_sample=None, dim=-1, 
                      filename='myimage', nrow=2, 
                      normalize=False):
    if id_sample is None:
        images = img_list[0:n_sample,:,:,:]
    else:
        images = img_list[id_sample,:,:,:]
    if dim >= 0:
        images = images[:,dim,:,:].unsqueeze(1)
    vutils.save_image(images, 
        '%s/%s'% (output_dir, filename+'.png'),
        nrow=nrow, normalize = normalize, padding=2)

# sampe iamges
def betterVisualizeAsImages(img_list, output_dir, 
                            n_sample=4, id_sample=None, dim=-1, 
                            filename='myimage', nrow=2, 
                            normalize=False, 
                            mask_list = None, useBound = False, upperBound = [0.5, 0.5, 2.2], lowerBound = [-0.5, -0.5, 1.2]):
    if id_sample is None:
        images = img_list[0:n_sample,:,:,:]
        if mask_list is not None:
            masks = mask_list[0:n_sample,:,:,:]
    else:
        images = img_list[id_sample,:,:,:]
        if mask_list is not None:
            masks = mask_list[id_sample,:,:,:]
    if useBound:
        for i in range(images.size(1)):
            images[:,i,:,:] = (images[:,i,:,:]-lowerBound[i])/(upperBound[i] - lowerBound[i])
    if mask_list is not None:
        images = torch.mul(images,masks)
    if dim >= 0:
        images = images[:,dim,:,:].unsqueeze(1)
    vutils.save_image(images, 
        '%s/%s'% (output_dir, filename+'.png'),
        nrow=nrow, normalize = normalize, padding=2)


def parseSampledDataPoint(dp0_img,nc):
    ###
    dp0_img  = dp0_img.float()/255 # convert to float and rerange to [0,1]
    #print(dp0_img.size())
    if nc==1:
        dp0_img  = dp0_img.unsqueeze(3)
    #print(dp0_img.size())
    dp0_img  = dp0_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    return dp0_img

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)    

def setCuda(*args):
    barg = []
    for arg in args: 
        barg.append(arg.cuda())
    return barg

def setAsVariable(*args):
    barg = []
    for arg in args: 
        barg.append(Variable(arg))
    return barg    

def getUpatedWeight(w, w2, dw, l1, l2, tau):
    w =  (1-tau)*w + tau*(dw*l1/l2)*w2
    return w

# ---- The model ---- #
# get the model definition/architecture
# get network
import WaspNetPerspective as WaspNet
encoders      = WaspNet.EncodersPerspectiveBCWarperCamera(opt)
decoders      = WaspNet.DecodersPerspectiveDepthRenderer2(opt)
if opt.cuda:
    encoders.cuda(device_id=0)
    decoders.cuda(device_id=0)

if not opt.modelPath=='':
    # rewrite here
    print('Reload previous model at: '+ opt.modelPath)
    encoders.load_state_dict(torch.load(opt.modelPath+'_encoders.pth'))
    decoders.load_state_dict(torch.load(opt.modelPath+'_decoders.pth'))
else:
    print('No previous model found, initializing model weight.')
    encoders.apply(weights_init)
    decoders.apply(weights_init)

print(opt.gpu_ids)
updator_encoders     = optim.Adam(encoders.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))
updator_decoders     = optim.Adam(decoders.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))


################    logs    ################ 
#logger = Logger(opt.dirLogger)         # doTensorBoardLog = False

################ criterion ################
criterionMSE1 = nn.MSELoss()
criterionABS2 = nn.L1Loss()
criterionMaskedABS1 = WaspNet.MaskedABSLoss()
criterionMaskedABS2 = WaspNet.MaskedABSLoss()
criterionMaskedMSE1 = WaspNet.MaskedMSELoss()
criterionMaskedMSE2 = WaspNet.MaskedMSELoss()
criterionAdv = nn.BCELoss()
criterionRecon = nn.L1Loss()
criterionSpWarp = nn.L1Loss()
criterionTVWarp = WaspNet.TotalVaryLoss(opt)
criterionBiasReduce = WaspNet.BiasReduceLoss(opt)
criterionPPL2Reg = WaspNet.PPL2Reg(opt)
criterionZDL2reg = WaspNet.L2Reg(opt)
#criterionAffL2Reg = nn.MSELoss()

if opt.cuda:
    criterionAdv = criterionAdv.cuda(device_id=0)

subject_ids  = ['006', '007', '008', '010', '013', '014', '017', '020', '021', '027']
label_in_use = [0,1,2,3,4,5,6,7,8,9]

# Training data folder list
TrainingData = {}
TestingData={}
label_space = {}
TrainingData[0] = []
TrainingData[0].append(opt.data_dir_prefix)


# ------------ training ------------ #
doTraining = True
doTesting = False
doTensorBoardLog = False            
doVanillaLog = True
doDynamicWeighting = False

iter_mark=0
w_deco = 0.1
w_deco_expect = w_deco
w_tvw = 1e-8
w_br = 0.0001
w_co = 0.15
w_ppl2 = 0.0001
w_ppl2_1 = 0.1*w_ppl2
w_ppl2_2 = 0.01*w_ppl2
w_zdl2 = 0.00

current_lr  = opt.lr

for epoch in range(opt.epoch_iter):
    train_loss = 0
    train_amount = 0+1e-6
    gc.collect() # collect garbage
    encoders.train()
    decoders.train()
    # create a set, key is label, value is the subset path
    TrainingDataSelection={}
    for iter_subset in range(3):
        if not doTraining:
            break
        for label, subsets in TrainingData.items():
            if label in label_in_use:
                TrainingDataSelection[label]=random.sample(subsets,1)[0]
        #dataset = WaspDataLoader.WaspImageFolderPair(root=dataroot)
        dataset = WaspDataLoader.WaspImageFolderMultilabelPairWithinlabels(info_dict=TrainingDataSelection)
        print('# size of the current (sub)dataset is %d' %len(dataset))
        train_amount = train_amount + len(dataset)
        num_batches  = math.ceil(len(dataset)/opt.batchSize)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
        for batch_idx, data_point in enumerate(dataloader, 0):
            #raw_input("Press Enter to continue...")
            gc.collect() # collect garbage
            ### prepare data ###
            dp0_img, dp0_label, dp9_img, dp9_label = data_point
            dp0_img =  parseSampledDataPoint(dp0_img, opt.nc)
            baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = dp0_img.size()[0])
            ones_vec = torch.cuda.FloatTensor(1).fill_(1).expand((dp0_img.size(0),1))
            ones_mat = torch.cuda.FloatTensor(opt.imgSize, opt.imgSize).fill_(1).expand((dp0_img.size(0), 1, opt.imgSize, opt.imgSize))
            zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
            zeroParams = torch.cuda.FloatTensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])     # This indicates no transformation - frontal pose
            if opt.cuda:
                zeroParams.cuda()
                dp0_img, baseg, zeroWarp = setCuda(dp0_img, baseg, zeroWarp)
            dp0_img = Variable(dp0_img)
            baseg = Variable(baseg, requires_grad=False)
            zeroWarp = Variable(zeroWarp, requires_grad=False)
            ones_vec = Variable(ones_vec, requires_grad=False)
            ones_mat = Variable(ones_mat, requires_grad=False)
            zeroParams = Variable(zeroParams, requires_grad=False)

            ###
            updator_encoders.zero_grad()
            updator_decoders.zero_grad()
            encoders.zero_grad()
            decoders.zero_grad()
            criterionRecon.zero_grad()
            criterionBiasReduce.zero_grad()
            criterionPPL2Reg.zero_grad()
            ### forward training points: dp0
            # Output is z, zimg, zdepth, zwarpimg, zwarpdepth, zpp
            dp0_z, dp0_zI, dp0_zD, dp0_zP = encoders(dp0_img)
            dp0_I, dp0_output, dp0_D, dp0_ppars = decoders(dp0_zI, dp0_zD, dp0_zP, baseg, ones_mat, ones_vec, zeroParams)
#            print('dp0_ppars[0:2,:] - ', dp0_ppars[0:2,:])
#            print('dp0_afpars[0,:] - ', dp0_afpars.data[0:10,:].transpose(0,1))

            ############## Encoder Decoder  training #####################
            updator_encoders.zero_grad()
            updator_decoders.zero_grad()
            encoders.zero_grad()
            decoders.zero_grad()
            criterionRecon.zero_grad()
            criterionTVWarp.zero_grad()
            criterionBiasReduce.zero_grad()
            criterionPPL2Reg.zero_grad()
            criterionZDL2reg.zero_grad()
            # reconstruction loss
            loss_recon = criterionRecon(dp0_output, dp0_img)
            loss_recon.backward(retain_graph=True)
            # l2 loss on affine transform: We regularise it so that the overall transformation is small.
            loss_ppl2 = criterionPPL2Reg(dp0_ppars, None, w1=w_ppl2_1, w2=w_ppl2_2)
#            loss_ppl2 = criterionPPL2Reg(dp0_afpars, zeroParams)
            loss_ppl2.backward(retain_graph=True)
#            loss_ppl2 = 0
            # smooth warping loss

            updator_decoders.step()
            updator_encoders.step()

            loss_encdec = loss_recon.data[0] + loss_ppl2.data[0]

            train_loss += loss_encdec
            #display training error and gradient
            iter_mark+=1
            print('Iteration[%d] -- EncDec loss -- ed:  %.4f .. ed_recon:  %.4f .. ed_ppl2: %.4f' 
                % (iter_mark, loss_encdec, loss_recon.data[0], loss_ppl2.data[0]),
                torch.mean(dp0_ppars,dim=0).data.cpu().numpy())

            # Save the second-last batch, for VanillaLogging. Sometimes, 
            # the last batch does not have enough images, so the log images
            # do not save as many images as we want. 
            if int(batch_idx) == int(num_batches - 2):
                save_dp0_img    = dp0_img.data.clone()
                save_dp0_output = dp0_output.data.clone()
                save_dp0_I      = dp0_I.data.clone()
                save_dp0_D      = dp0_D.data.clone()
            info = {
                    'loss_encdec': loss_encdec,
                    'loss_recon': loss_recon.data[0],
                    'loss_ppl2': loss_ppl2.data[0],
            }
            
            if doVanillaLog:
                # Do vanilla logging into a file. 
                vanillaLog.logInfo(opt.vanillaLogFile, info, iter_mark)

            ################# TensorBoard logging ################# 
            if doTensorBoardLog and (iter_mark+1) % 40 == 0:
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, iter_mark+1)
                # (2) Log values and gradients of the parameters (histogram)
                #for tag, value in net.named_parameters():
                #    tag = tag.replace('.', '/')
                #    logger.histo_summary(tag, to_np(value), step+1)
                #    logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)
                # (3) Log the images
                info2 = {
                    'dp0_img': to_np(dp0_img.view(-1, 64, 64)[:4]),
                    'dp0_output': to_np(dp0_output.view(-1, 64, 64)[:4]),
                    'dp0_I': to_np(dp0_I.view(-1, 64, 64)[:4])
                }
                for tag, images in info2.items():
                    logger.image_summary(tag, images, iter_mark+1)

        ################# Vanilla logging #################
        if doVanillaLog:
#            gx = (dp0_W.data[:,0,:,:]+baseg.data[:,0,:,:]).unsqueeze(1).clone()
#            gy = (dp0_W.data[:,1,:,:]+baseg.data[:,1,:,:]).unsqueeze(1).clone()
            # save images after iterating a dataroot
            # save images after iterating a dataroot
#            visualizeAsImages(dp0_img.data.clone(), 
            visualizeAsImages(save_dp0_D,
                opt.dirImageoutput, 
                filename='iter_'+str(iter_mark)+'_depth0_', n_sample = 49, nrow=7, normalize=False) 
            visualizeAsImages(save_dp0_img,
                opt.dirImageoutput, 
                filename='iter_'+str(iter_mark)+'_img0_', n_sample = 49, nrow=7, normalize=False) 
#            visualizeAsImages(dp0_I.data.clone(), 
            visualizeAsImages(save_dp0_I,
                opt.dirImageoutput, 
                filename='iter_'+str(iter_mark)+'_tex0_', n_sample = 49, nrow=7, normalize=False)
#            visualizeAsImages(dp0_output.data.clone(), 
            visualizeAsImages(save_dp0_output,
                opt.dirImageoutput, 
                filename='iter_'+str(iter_mark)+'_output0_', n_sample = 49, nrow=7, normalize=False)   
            #raw_input("Press Enter to continue...")
            np.save('%s/iter_%d_depth0_.npy' %(opt.dirImageoutput, iter_mark), save_dp0_D.cpu().numpy())
            np.save('%s/iter_%d_tex0_.npy' %(opt.dirImageoutput, iter_mark), save_dp0_img.cpu().numpy())

    # Reduce learning rate after one pass through training data.   
    current_lr          = 0.8*current_lr
    print('Reducing learnign rate. New: %.6f' %(current_lr))
    updator_encoders    = optim.Adam(encoders.parameters(), lr = current_lr, betas=(opt.beta1, 0.999))
    updator_decoders    = optim.Adam(decoders.parameters(), lr = current_lr, betas=(opt.beta1, 0.999))

    if doTraining:
        print('====> Epoch: {} Average training loss: {:.4f}'.format(epoch, train_loss / train_amount))
        # do checkpointing
        torch.save(encoders.state_dict(), '%s/wasp_model_epoch_%d_encoders.pth' % (opt.dirCheckpoints, epoch))
        torch.save(decoders.state_dict(), '%s/wasp_model_epoch_%d_decoders.pth' % (opt.dirCheckpoints, epoch))

    # ------------ testing ------------ #
    #model.eval()
    test_amount = 0+1e-6
    test_loss = 0
    iter_mark_testing=0
    
    # on synthetic image set
    print('Testing on synthetic images ... ')
    #raw_input("Press Enter to continue...")
    gc.collect() # collect garbage
    for dataroot in TestingData:
        if not doTesting:
            break
        dataset = WaspDataLoader.FareDataSet(root=dataroot)
        print('# size of the current (sub)dataset is %d' %len(dataset))
        test_amount = test_amount + len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
        for batch_idx, data_point in enumerate(dataloader, 0):
            gc.collect() # collect garbage
            # unpack the loaded and processed data to whatever it is    
            img0, mcc0= data_point
            dp0_img, dp0_cc, dp0_cc_z, dp0_mask = parseSampledDataPoint(img0, mcc0, getDepth=True)

            
            dp0_mask3 = dp0_mask.repeat(1,3,1,1)
            dp0_mask3_float = torch.FloatTensor(dp0_mask3.size()).copy_(dp0_mask3)
            dp0_mask_float = torch.FloatTensor(dp0_mask.size()).copy_(dp0_mask)

            if opt.cuda:
                dp0_img, dp0_cc, dp0_cc_z, dp0_mask3= setCuda(dp0_img, dp0_cc, dp0_cc_z, dp0_mask3)
                dp0_mask3_float, dp0_mask_float = setCuda(dp0_mask3_float, dp0_mask_float)

            dp0_img, dp0_cc, dp0_cc_z, dp0_mask3_float, dp0_mask_float = setAsVariable(dp0_img, dp0_cc, dp0_cc_z, dp0_mask3_float, dp0_mask_float)
            optimizer.zero_grad()

            ############## Train U ###############
            model.zero_grad()
            makeCC.zero_grad()

            # model regresses img0 to cc0
            pred_cc_z_dr = model(dp0_img)
            pred_cc_z = RerangeZ(pred_cc_z_dr)
            # regression/reconstruction loss
            loss_recon = criterionMaskedABS1(pred_cc_z, dp0_cc_z, dp0_mask3_float)
            # get camera coord prediction by stacking z to x and y
            pred_cc = makeCC(pred_cc_z)
            # get gt mask associated with this warping
            test_loss +=  loss_recon.data[0]
            #display training error and gradient
            iter_mark_testing += 1
            print('Testing Iteration[%d] ... Testing error: %.4f ... ' 
                % (iter_mark_testing, loss_recon.data[0]))
        visualizeAsImages(dp0_img.data.clone(), 
            opt.dirTestingoutput+'/synthetic', 
            filename='iter_'+str(iter_mark)+'_testIter_'+str(iter_mark_testing)+'_img0_', n_sample = 4, nrow=2, normalize=False)
        betterVisualizeAsImages(pred_cc.data.clone(), 
            opt.dirTestingoutput+'/synthetic', 
            filename='iter_'+str(iter_mark)+'_testIter_'+str(iter_mark_testing)+'_pred0_', n_sample = 4, nrow=2, normalize=False,
            mask_list = dp0_mask3_float.data, useBound = True, upperBound = [0.5, 0.5, 2.2], lowerBound = [-0.5, -0.5, 1.2])
        betterVisualizeAsImages(pred_cc.data.clone(), 
            opt.dirTestingoutput+'/synthetic', 
            filename='iter_'+str(iter_mark)+'_testIter_'+str(iter_mark_testing)+'_pred0z_', dim =2, n_sample = 4, nrow=2, normalize=False,
            mask_list = dp0_mask3_float.data, useBound = True, upperBound = [0.5, 0.5, 2.2], lowerBound = [-0.5, -0.5, 1.2])
        betterVisualizeAsImages(dp0_cc.data.clone(), 
            opt.dirTestingoutput+'/synthetic', 
            filename='iter_'+str(iter_mark)+'_testIter_'+str(iter_mark_testing)+'_cc0_', n_sample = 4, nrow=2, normalize=False,
            mask_list = dp0_mask3_float.data, useBound = True, upperBound = [0.5, 0.5, 2.2], lowerBound = [-0.5, -0.5, 1.2])
        betterVisualizeAsImages(dp0_cc.data.clone(), 
            opt.dirTestingoutput+'/synthetic', 
            filename='iter_'+str(iter_mark)+'_testIter_'+str(iter_mark_testing)+'_cc0z_', dim = 2, n_sample = 4, nrow=2, normalize=False,
            mask_list = dp0_mask3_float.data, useBound = True, upperBound = [0.5, 0.5, 2.2], lowerBound = [-0.5, -0.5, 1.2])
        betterVisualizeAsImages(torch.abs(pred_cc.data-dp0_cc.data).clone(), 
            opt.dirTestingoutput+'/synthetic', 
            filename='iter_'+str(iter_mark)+'_testIter_'+str(iter_mark_testing)+'_error_', dim =2, n_sample = 4, nrow=2, normalize=False,
            mask_list = dp0_mask3_float.data, useBound = False)
        visualizeAsImages(pred_cc_z_dr.data.clone(), 
            opt.dirImageoutput, 
            filename='iter_'+str(iter_mark)+'_testIter_'+str(iter_mark_testing)+'_pred_cc_z_dr', n_sample = 4, nrow=2, normalize=False)     
    print('====> Epoch: {} Average testing error: {:.4f}'.format(epoch, test_loss / test_amount))


    gc.collect() # collect garbage






























    ##
