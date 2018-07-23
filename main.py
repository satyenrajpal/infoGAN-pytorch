from trainer import Trainer
import torch
import os
import argparse
# TODO:
#  - Replace BatchNorm with Instance Norm WORKS!
#  - L2 loss instead of factored Gaussian!?
#  - Replicate DC GAN architecture - Not working!

def str2bool(v):
    return v.lower() in ('true')

def main(config):

    if config.mode == 'train':
        
        config.sample_save_dir = os.path.join(config.save_dir, 'samples')
        if not os.path.exists(config.sample_save_dir):
            os.makedirs(config.sample_save_dir)
        
        config.model_save_dir = os.path.join(config.save_dir, 'models')
        if not os.path.exists(config.model_save_dir):
            os.makedirs(config.model_save_dir)
        
        trainer = Trainer(config)
        
        if config.restore_dir!='':
            trainer.restore_models(config.restore_dir,config.resume_epoch,config.resume_iter)
        
        print("Training...")
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--dataset', type=str,default='MNIST',help='Which dataset to train/test model',choices=['MNIST'])
    parser.add_argument('--num_d', type=int, default=10, help='number of domain labels')
    parser.add_argument('--num_c', type=int, default=2, help='number of continuous dimensions')
    parser.add_argument('--dim_z', type=int, default=30, help='dimension of noise vector')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_gp', type=float, default=5, help='weight for gradient penalty')
    parser.add_argument('--lambda_MI', type=float, default=1, help='weight for Mutual information loss')
    parser.add_argument('--image_size', type=float, default=32, help='Size of images in dataset')
    parser.add_argument('--FE_conv_dim',type=int,default=64,help='start conv dim for FrontEnd')
    parser.add_argument('--g_conv_dim',type=int,default=1024, help = 'start conv dim for Generator')
    parser.add_argument('--crop_size',type=int,default=650,help='Crop Size for RafD dataset')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iter', type=int,default=0)
    parser.add_argument('--resume_epoch', type=int,default=0)
    
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test','calc_score'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--pre_rafd', action='store_true', help='Preprocess RafD?')

    # Directories.
    parser.add_argument('--mnist_dir', type=str, default='dataset')
    parser.add_argument('--rafd_image_dir', type=str, default='../RafD')
    parser.add_argument('--save_dir', type=str, default='cinfoResults')
    parser.add_argument('--restore_dir',type=str,default='',help='Directory containing models to be loaded')

    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    # print(config)
    main(config)