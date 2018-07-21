import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
from data_loader import get_loader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np
import sys
from model import FrontEnd,D,Q,G,weights_init

class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

class Trainer:

    def __init__(self, config):

        #Essential Training configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.batch_size
        self.num_d = config.num_d
        self.num_c = config.num_c
        self.channels = 1 if config.dataset.lower() == 'mnist' else 3
        self.image_size = config.image_size
        self.num_epochs = config.num_epochs
        self.dataset = config.dataset

        # Directories
        self.sample_save_dir = config.sample_save_dir
        self.model_save_dir = config.model_save_dir
        if self.dataset.lower() == 'mnist':
            self.image_dir = config.mnist_dir
    
        # Hyperparameters
        self.lambda_cls = config.lambda_cls
        self.lambda_MI = config.lambda_MI
        self.lambda_gp = config.lambda_gp
        self.lambda_con = config.lambda_con
        self.n_critic = config.n_critic
        self.FE_conv_dim = config.FE_conv_dim
        self.g_conv_dim = config.g_conv_dim
        
        # Misc
        self.num_workers = config.num_workers
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.log_step = config.log_step
        self.build_models()
    
    def build_models(self):
        self.FE = FrontEnd(self.channels, conv_dim=self.FE_conv_dim, image_size=self.image_size)
        self.D = D(classes = self.num_d, FE_dim = self.FE.end_dim)
        self.Q = Q(output_c = self.num_c, FE_dim = self.FE.end_dim)
        self.G = G(input_ = self.num_d+self.num_c, g_conv_dim=self.g_conv_dim, image_size=self.image_size,output_c=self.channels)
    
        self.FE.to(self.device).apply(weights_init)
        self.D.to(self.device).apply(weights_init)
        self.Q.to(self.device).apply(weights_init)
        self.G.to(self.device).apply(weights_init)

    def save_models(self,epoch,num_iters):
        G_path  = os.path.join(self.model_save_dir,'{}-{}-G.ckpt'.format(epoch,num_iters))
        FE_path = os.path.join(self.model_save_dir,'{}-{}-FE.ckpt'.format(epoch,num_iters))
        D_path  = os.path.join(self.model_save_dir,'{}-{}-D.ckpt'.format(epoch,num_iters))
        Q_path  = os.path.join(self.model_save_dir,'{}-{}-Q.ckpt'.format(epoch,num_iters))
        torch.save(self.G.state_dict(),G_path)
        torch.save(self.FE.state_dict(),FE_path)
        torch.save(self.D.state_dict(),D_path)
        torch.save(self.Q.state_dict(),Q_path)

    def restore_models(self,dir_,epoch,iters):
        print("Loading the trained models from {} epoch and {} step in {} dir".format(epoch,iters,dir_))
        G_path  = os.path.join(dir_,'{}-{}-G.ckpt'.format(epoch,iters))
        FE_path = os.path.join(dir_,'{}-{}-FE.ckpt'.format(epoch,iters))
        D_path  = os.path.join(dir_,'{}-{}-D.ckpt'.format(epoch,iters))
        Q_path  = os.path.join(dir_,'{}-{}-Q.ckpt'.format(epoch,iters))
        self.G.load_state_dict(torch.load(G_path,map_location= lambda storage, loc: storage))
        self.FE.load_state_dict(torch.load(FE_path,map_location= lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path,map_location= lambda storage, loc: storage))
        self.Q.load_state_dict(torch.load(Q_path,map_location= lambda storage, loc: storage))


    def _make_conditions(self,labels,x_label,con_c,bs):

        c = np.zeros((bs, 10))
        c[range(bs),x_label] = 1.0
        labels.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0,1.0)
        z = torch.cat([labels,con_c],dim=1).view(bs,-1,1,1)

        return z

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def optLosses(self):
        self.CELoss = nn.CrossEntropyLoss().to(self.device)
        self.GaussLoss = log_gaussian() 
        self.optimD = optim.Adam([{'params' : self.FE.parameters()}, {'params' : self.D.parameters()}], lr = 0.0002, betas = (0.5, 0.99))
        self.optimG = optim.Adam([{'params' : self.G.parameters() }, {'params' : self.Q.parameters()}], lr = 0.0001, betas = (0.5, 0.99))
    
    #CHANGE DIM OF THIS!
    def make_fixed_cond(self):
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1,1)

        con_c_ = []
        for i in range(self.num_c):
            p = np.zeros((100,self.num_c))
            p[:,i] = np.squeeze(c)
            con_c_.append(p)
        
        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        fix_labels = torch.FloatTensor(torch.from_numpy(one_hot).float()).to(self.device)
        fix_con_c = torch.FloatTensor(100,self.num_c).to(self.device)
        
        return con_c_, fix_con_c, fix_labels

    def train(self):

        dataloader = get_loader(self.image_size,self.batch_size,self.image_dir,self.num_workers,self.dataset)
        self.optLosses()

        real_x   = torch.FloatTensor(self.batch_size, self.channels, self.image_size, self.image_size).requires_grad_().to(self.device)
        rf_label = torch.FloatTensor(self.batch_size).to(self.device)
        labels   = torch.FloatTensor(self.batch_size, self.num_d).requires_grad_().to(self.device)
        con_c    = torch.FloatTensor(self.batch_size, self.num_c).requires_grad_().to(self.device)
        
        
        # fixed random variables for testing
        con_c_, fix_con_c, fix_labels = self.make_fixed_cond()

        
        for epoch in range(self.num_epochs):
          for num_iters, batch_data in enumerate(dataloader, 0):

            # real part
            self.optimD.zero_grad()
            
            x, x_label = batch_data 
            
            bs = x.size(0)
            real_x.data.resize_(x.size())
            rf_label.data.resize_(bs,1)
            labels.data.resize_(bs, self.num_d)
            con_c.data.resize_(bs, self.num_c)
            
            # Random conditioned 'z' 
            cond = self._make_conditions(labels, x_label, con_c, bs)        
            x_label = x_label.to(self.device)
            
            # Real part
            real_x.data.copy_(x)
            fe_out1 = self.FE(real_x)
            out_real, out_cls = self.D(fe_out1)
            
            # GAN loss
            loss_real = -torch.mean(out_real)
            
            # Classification loss
            loss_class = self.CELoss(out_cls,x_label)
            
            # fake part
            fake_x = self.G(cond)
            fe_out2 = self.FE(fake_x.detach())
            out_fake, _ = self.D(fe_out2)
            loss_fake = torch.mean(out_fake)
            
            alpha = torch.rand(real_x.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * real_x.data + (1 - alpha) * real_x.data).requires_grad_(True)
            out_src, _ = self.D(self.FE(x_hat))
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            D_loss = loss_real + loss_fake + self.lambda_cls* loss_class + self.lambda_gp*d_loss_gp
            D_loss.backward()
            self.optimD.step()

            # G and Q part
            if (num_iters+1) % self.n_critic ==0:
                self.optimG.zero_grad()

                x_fake = self.G(cond)
                fe_out = self.FE(fake_x)
                out_fake, out_cls = self.D(fe_out)
                
                # WGAN loss
                fake_loss = -torch.mean(out_fake)
                
                #Classification loss
                g_loss_cls = self.CELoss(out_cls,x_label)

                q_mu, q_var = self.Q(fe_out)
                con_loss = self.GaussLoss(con_c, q_mu, q_var)
                
                G_loss = fake_loss + self.lambda_cls*g_loss_cls + self.lambda_con*con_loss
                G_loss.backward()
                self.optimG.step()

            if (num_iters+1) % self.log_step ==0:
                print('Epoch - Iter:{0} - {1}, Dloss: {2}, Gloss: {3}, Classification Loss: {4}, Gaussian Loss: {5}'.format(
                       epoch, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy(),
                        loss_class.data.cpu().numpy(),
                        con_loss.data.cpu().numpy()))

            if (num_iters+1) % self.sample_step == 0:
                
                with torch.no_grad():
                    for i,cont_c in enumerate(con_c_):
                        fix_con_c.data.copy_(torch.from_numpy(cont_c))
                        z = torch.cat([fix_labels, fix_con_c], 1).view(100, -1 , 1, 1)
                        x_save = self.G(z)
                        save_image(x_save.data.cpu(), self.sample_save_dir + '/{}-{}-c{}.png'.format(epoch,num_iters,i), nrow=10)

            if (num_iters+1) % self.model_save_step:
                self.save_models(epoch, num_iters)