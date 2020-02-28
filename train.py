from conv_rbm import *
from DataLoader import *

import torchvision.utils as vutils

from attr_dict import make_recursive_attr_dict
import yaml

import argparse
import pickle
import os

import sys

def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()
    return

optim                           = torch.optim

def main(sys_string=None):
    parser                      = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='', type=str, help='Path to configuration file.')
    parser.add_argument('--gpu_id', default=0, type=int, help='Which GPU to use.')
    if not sys_string:
        args                    = parser.parse_args()
    else:
        args                    = parser.parse_args(sys_string.split(' '))

    torch.cuda.set_device(args.gpu_id)

    exp_name                    = os.path.split(args.cfg)[1].replace('.yaml', '')

    output_path                 = os.path.join('output/', exp_name)

    for odir in ['images/input/', 'images/hidden/', 'images/reconstruction/', 'images/weights/', 'images/hbias/']:
        if not os.path.exists(os.path.join(output_path, odir)):
            os.makedirs(os.path.join(output_path, odir))

    with open(args.cfg, 'r') as fp:
        options                 = yaml.safe_load(fp)

    options                     = make_recursive_attr_dict(options)
    options                     = fix_backward_compatibility(options)

    model                       = ConvRBM(options)

    if options.training.cuda:
        model.cuda()
        
#    optimiser                   = optim.Adam(
#                                    model.parameters(), 
#                                    lr=options.optimiser.lr, 
#                                    betas=(options.optimiser.beta1, options.optimiser.beta2),
#                                    weight_decay=options.training.weight_decay,
#                                  )
    optimiser                   = optim.SGD(
                                    model.parameters(),
                                    lr=options.optimiser.lr,
                                    momentum=0,
                                    weight_decay=0
                                  )

    dataloader                  = DataLoader(options.training.dataset_file)

    loss_recon_history          = []
    loss_sparsity_history       = []
    iter_mark                   = 0
    epoch                       = 0

    wid                         = 0

    def save_model(output_path='output/foo/'):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        torch.save(model.state_dict(), os.path.join(output_path, 'model.pth'))
        torch.save(optimiser.state_dict(), os.path.join(output_path, 'optimiser.pth'))
        with open(os.path.join(output_path, 'options.pkl'), 'wb') as fp:
            pickle.dump(options, fp)
        
        system_state            = {
            'loss_recon_history'    : loss_recon_history,
            'loss_sparsity_history' : loss_sparsity_history,
            'iter_mark'             : iter_mark,
            'epoch'                 : epoch,
        }
        with open(os.path.join(output_path, 'system_state.pkl'), 'wb') as fp:
            pickle.dump(system_state, fp)

    def load_model(path='output/foo/'):
        if not os.path.exists(path):
            print('%s does not exist!' %(path))
            raise ValueError
        model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        optimiser.load_state_dict(torch.load(os.path.join(path, 'optimiser.pth')))
        with open(os.path.join(path, 'system_state.pkl'), 'rb') as fp:
            system_state        = pickle.load(fp)
        loss_recon_history      = system_state['loss_recon_history']
        loss_sparsity_history   = system_state['loss_sparsity_history']
        iter_mark               = system_state['iter_mark']
        epoch                   = system_state['epoch']

    if options.training.load_model is not None:
        print('Resuming from old model at %s.' %(options.training.load_model))
        load_model(options.training.load_model)

    model.set_lr(epoch)

    while iter_mark < options.training.n_iter:
#        optimiser.zero_grad()
        model.reset_grad()
        model.set_momentum(iter_mark)

        X, last_batch           = dataloader.next_batch(options.training.batch_size, options.training.patch_size)
        X                       = X - X.mean()

        if options.training.cuda:
            X                   = X.cuda()

        X_hat                   = model(X)

        loss_total              = model.compute_losses()
        model.compute_updates()
        model.update()

        loss_recon              = model.loss_recon
        loss_sparsity           = model.loss_sparsity

        loss_recon_history.append(loss_recon.item())
        loss_sparsity_history.append(loss_sparsity.item())
        

        write_flush('\r' + ' ' * 150 + '\rIteration %5d . LR = %.4g | recon: %.4f . sparsity: %.4f . momentum = %.4f . std_gaussian = %.4f ; ||W|| = %.4f . ||bh|| = %.4f . ||bv|| = %.4f . ||dW|| = %.4f . ||dHbias|| = %.4f . mean(Hprobs) = %.4f' 
                %(iter_mark, model.lr, loss_recon.item(), loss_sparsity.item(), model.momentum, model.std_gaussian,
                  torch.norm(model.W), torch.norm(model.bh), torch.norm(model.bv), torch.norm(model.W.grad), torch.norm(model.bh.grad), model.Hprobs0.mean()))
        if options.model.use_vbias:
            write_flush(' .||dVbias|| = %.4f' %(torch.norm(model.bv.grad)))

#        optimiser.step()

        if iter_mark % 500 == 0:
            write_flush('\n')

        if (iter_mark + 1) % options.checkpoint.step == 0:
            save_model(output_path=output_path)

            vutils.save_image(
                    X,
                    os.path.join(output_path, 'images/input/%06d.png' %(iter_mark)),
                    nrow=int(np.floor(np.sqrt(options.training.batch_size))),
                    normalize=True,
            )
            vutils.save_image(
                    X_hat, 
                    os.path.join(output_path, 'images/reconstruction/%06d.png' %(iter_mark)),
                    nrow=int(np.floor(np.sqrt(options.training.batch_size))),
                    normalize=True,
            )
            vutils.save_image(
                    model.Hprobs0[0,:,:,:].unsqueeze(1), 
                    os.path.join(output_path, 'images/hidden/%06d.png' %(iter_mark)),
                    nrow=int(np.floor(np.sqrt(options.model.num_weights))),
                    normalize=True,
            )

        if last_batch:
            vutils.save_image(
                model.W,
                os.path.join(output_path, 'images/weights/%06d.png' %(epoch)),
                nrow=int(np.floor(np.sqrt(options.model.num_weights))),
                padding=2,
                normalize=True,
            )
            torch.save(
                model.W,
                os.path.join(output_path, 'images/weights/%06d.pth' %(epoch))
            )
            torch.save(
                model.bh,
                os.path.join(output_path, 'images/hbias/%06d.pth' %(epoch))
            )

            wid                += 1
        
            epoch              += 1
            model.set_lr(epoch)


        iter_mark              += 1

    write_flush('\n')

if __name__ == '__main__':
    main()
