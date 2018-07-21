import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import numpy as np
import sys

class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

class Trainer:

    def __init__(self, G, FE, D, Q,num_c=2,num_d=10):

        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 100
        self.num_c = num_c
        self.num_d = num_d
        self.n_critic = 1

    def _noise_sample(self, dis_c, con_c, noise, bs):

        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs),idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

        return z, idx

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

    def train(self):

        if not os.path.exists(os.path.join(os.getcwd(),'samples')):
            os.makedirs('samples')

        real_x   = torch.FloatTensor(self.batch_size, 1, 28, 28).requires_grad_().to(self.device)
        rf_label = torch.FloatTensor(self.batch_size).to(self.device)
        labels   = torch.FloatTensor(self.batch_size, self.num_d).requires_grad_().to(self.device)
        con_c    = torch.FloatTensor(self.batch_size, self.num_c).requires_grad_().to(self.device)
        
        dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(),download=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        criterionD = nn.BCELoss().to(self.device)
        criterionQ_dis = nn.CrossEntropyLoss().to(self.device)
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params' : self.FE.parameters()}, {'params' : self.D.parameters()}], lr = 0.0002, betas = (0.5, 0.99))
        optimG = optim.Adam([{'params' : self.G.parameters() }, {'params' : self.Q.parameters()}], lr = 0.0001, betas = (0.5, 0.99))

        # fixed random variables for testing
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1,1)

        con_c_ = []
        for i in range(self.num_c):
            p = np.zeros((self.batch_size,self.num_c))
            p[:,i] = np.squeeze(c)
            con_c_.append(p)

        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        # fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)

        for epoch in range(100):
          for num_iters, batch_data in enumerate(dataloader, 0):

            # real part
            optimD.zero_grad()
            
            x, x_label = batch_data 
            
            bs = x.size(0)
            real_x.data.resize_(x.size())
            rf_label.data.resize_(bs,1)
            labels.data.resize_(bs, self.num_d)
            con_c.data.resize_(bs, self.num_c)
            # noise.data.resize_(bs, 62)
            
            # Random conditioned 'z' 
            cond = self._make_conditions(labels, x_label, con_c, bs)        
            x_label = x_label.to(self.device)
            
            # Real part
            real_x.data.copy_(x)
            fe_out1 = self.FE(real_x)
            out_real, out_cls = self.D(fe_out1)
            # rf_label.data.fill_(1.0)

            # GAN loss
            # loss_real = criterionD(probs_real, rf_label)
            loss_real = -torch.mean(out_real)
            loss_real.backward(retain_graph=True)

            # Classification loss
            loss_class = criterionQ_dis(out_cls,x_label)
            loss_class.backward()
            
            # fake part
            fake_x = self.G(cond)
            fe_out2 = self.FE(fake_x.detach())
            out_fake, _ = self.D(fe_out2)
            # rf_label.data.fill_(0)
            # loss_fake = criterionD(probs_fake, rf_label)
            loss_fake = torch.mean(out_fake)
            loss_fake.backward()

            alpha = torch.rand(real_x.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * real_x.data + (1 - alpha) * real_x.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            D_loss = loss_real + loss_fake + loss_class + 5*d_loss_gp

            optimD.step()

            # G and Q part
            if (num_iters+1) % self.n_critic ==0:
                optimG.zero_grad()

                x_fake = self.G(cond)
                fe_out = self.FE(fake_x)
                out_fake, out_cls = self.D(fe_out)
                # rf_label.data.fill_(1.0)
                
                # WGAN loss
                fake_loss = -torch.mean(out_fake)
                
                #Classification loss
                g_loss_cls = criterionQ_dis(out_cls,x_label)

                q_mu, q_var = self.Q(fe_out)
                # dis_loss = criterionQ_dis(q_logits, target)
                con_loss = criterionQ_con(con_c, q_mu, q_var)
                
                G_loss = fake_loss + g_loss_cls + con_loss
                G_loss.backward()
                optimG.step()

            if num_iters % 100 == 0:

              print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}, Classification Loss: {4}, Gaussian Loss: {5}'.format(
                epoch, num_iters, D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy(),
                reconstruct_loss.data.cpu().numpy(),
                con_loss.data.cpu().numpy()))

              # noise.data.copy_(fix_noise)
              labels.data.copy_(torch.Tensor(one_hot))

              for i,cont_c in enumerate(con_c_):
                con_c.data.copy_(torch.from_numpy(cont_c))
                z = torch.cat([labels, con_c], 1).view(bs, -1 , 1, 1)
                x_save = self.G(z)
                save_image(x_save.data, 'samples/{}-{}-c{}.png'.format(epoch,num_iters,i), nrow=10)

              