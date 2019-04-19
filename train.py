from conv_rbm import *
from DataLoader import *

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

def main():
    parser                      = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='', type=str, help='Path to configuration file.')
    parser.add_argument('--gpu_id', default=0, type=int, help='Which GPU to use.')
    args                        = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    exp_name                    = os.path.split(args.cfg)[1].replace('.yaml', '')

    output_path                 = os.path.join('output/', exp_name)

    with open(args.cfg, 'r') as fp:
        options                 = yaml.safe_load(fp)

    options                     = make_recursive_attr_dict(options)

    model                       = ConvRBM(
                                    options.model.weight_size,
                                    options.model.channels,
                                    options.model.num_weights,
                                    options.model.pool_size,
                                    options.model.sparsity,
                                    sigm=options.model.sigmoid,
                                    reuse_vbias=options.model.use_vbias,
                                    k_CD=options.model.k_CD
                                  )

    if options.training.cuda:
        model.cuda()
        
    optimiser                   = optim.Adam(
                                    model.parameters(), 
                                    lr=options.optimiser.lr, 
                                    betas=(options.optimiser.beta1, options.optimiser.beta2),
                                    weight_decay=options.training.weight_decay,
                                  )

    dataloader                  = DataLoader(options.training.dataset_file)

    loss_recon_history          = []
    loss_sparsity_history       = []
    iter_mark                   = 0

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
            'iter_mark'             : iter_mark
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

    if options.training.load_model != '':
        print('Resuming from old model at %s.' %(options.training.load_model))
        load_model(options.training.load_model)

    while iter_mark < options.training.n_iter:
        optimiser.zero_grad()

        X                       = dataloader.next_batch(options.training.batch_size, options.training.patch_size)

        if options.training.cuda:
            X                   = X.cuda()

        X_hat                   = model(X)

        loss_total              = model.compute_losses()
        model.compute_updates()

        loss_recon              = model.loss_recon
        loss_sparsity           = model.loss_sparsity

        loss_recon_history.append(loss_recon.item())
        loss_sparsity_history.append(loss_sparsity.item())
        
        optimiser.step()

        write_flush('\r' + ' ' * 100 + '\rIteration %5d . LR = %g | recon: %.4f . sparsity: %.4f' %(iter_mark, optimiser.param_groups[0]['lr'], loss_recon.item(), loss_sparsity.item()))

        if (iter_mark + 1) % 500 == 0:
            write_flush('\n')

        if (iter_mark + 1) % options.checkpoint.step == 0:
            save_model(output_path=output_path)

        if (iter_mark + 1) in options.training.lr_decay_step:
            for pg in optimiser.param_groups:
                pg['lr']       *= options.training.lr_decay

        iter_mark              += 1

    write_flush('\n')

if __name__ == '__main__':
    main()
