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

class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

class Trainer:

  def __init__(self, G, FE, D, Q):

    self.G = G
    self.FE = FE
    self.D = D
    self.Q = Q
    self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.batch_size = 100

  def _noise_sample(self, dis_c, con_c, noise, bs):

    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs),idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

    return z, idx

  def _make_conditions(self,labels_x,con_c,bs):

    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs),idx] = 1.0
    
    labels_x.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0,1.0)

    z = torch.cat([labels_x,con_c],dim=1).view(bs,-1,1,1)

    return z,idx

  def train(self):

    if not os.path.exists(os.path.join(os.getcwd(),'samples')):
        os.makedirs('samples')

    real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).to(self.device)
    rf_label = torch.FloatTensor(self.batch_size).to(self.device)
    labels_x = torch.FloatTensor(self.batch_size, 10).to(self.device)
    con_c = torch.FloatTensor(self.batch_size, 2).to(self.device)
    # noise = torch.FloatTensor(self.batch_size, 62).to(self.device)

    real_x = Variable(real_x)
    rf_label = Variable(rf_label, requires_grad=False)
    labels_x = Variable(labels_x)
    con_c = Variable(con_c)
    # noise = Variable(noise)

    dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(),download=True)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    criterionD = nn.BCELoss().to(self.device)
    criterionQ_dis = nn.CrossEntropyLoss().to(self.device)
    criterionQ_con = log_gaussian()

    optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))


    # fixed random variables for testing
    c = np.linspace(-1, 1, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])

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
        rf_label.data.resize_(bs)
        labels_x.data.resize_(bs, 10)
        con_c.data.resize_(bs, 2)
        # noise.data.resize_(bs, 62)
        
        # Random conditioned 'z' 
        z, idx = self._make_conditions(labels_x,con_c,bs)        
        class_ = torch.LongTensor(idx).to(self.device)
        target = Variable(class_)

        # Real part
        real_x.data.copy_(x)
        fe_out1 = self.FE(real_x)
        # print("FE output", fe_out1.size())
        probs_real, out_cls = self.D(fe_out1)
        rf_label.data.fill_(1)
        
        # GAN loss
        loss_real = criterionD(probs_real, rf_label)
        loss_real.backward(retain_graph=True)

        # Classification loss
        loss_class = criterionQ_dis(out_cls,class_)
        loss_class.backward()
        
        # fake part
        # z, idx = self._noise_sample(dis_c, con_c, noise, bs)
        fake_x = self.G(z)
        fe_out2 = self.FE(fake_x.detach())
        probs_fake,_ = self.D(fe_out2)
        rf_label.data.fill_(0)
        loss_fake = criterionD(probs_fake, rf_label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake + loss_class

        optimD.step()
        
        # G and Q part
        optimG.zero_grad()

        #PASS z THROUGH G AGAIN!?
        # x_fake = self.G(z)
        fe_out = self.FE(fake_x)
        probs_fake, out_cls = self.D(fe_out)
        rf_label.data.fill_(1.0)
        
        # GAN loss
        reconstruct_loss = criterionD(probs_fake, rf_label)
        
        #Classification loss
        g_loss_cls = criterionQ_dis(out_cls,class_)

        q_mu, q_var = self.Q(fe_out)
        # dis_loss = criterionQ_dis(q_logits, target)
        con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1
        
        G_loss = reconstruct_loss + g_loss_cls + con_loss
        G_loss.backward()
        optimG.step()

        if num_iters % 100 == 0:

          print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}, Classification Loss: {4}, Gaussian Loss: {5}'.format(
            epoch, num_iters, D_loss.data.cpu().numpy(),
            G_loss.data.cpu().numpy(),
            reconstruct_loss.data.cpu().numpy(),
            con_loss.data.cpu().numpy()))

          # noise.data.copy_(fix_noise)
          labels_x.data.copy_(torch.Tensor(one_hot))

          con_c.data.copy_(torch.from_numpy(c1))
          z = torch.cat([labels_x, con_c], 1).view(bs, -1 , 1, 1)
          x_save = self.G(z)
          save_image(x_save.data, 'samples/{}-{}-c1.png'.format(epoch,num_iters), nrow=10)

          con_c.data.copy_(torch.from_numpy(c2))
          z = torch.cat([labels_x, con_c], 1).view(bs, -1 , 1, 1)
          x_save = self.G(z)
          save_image(x_save.data, 'samples/{}-{}-c2.png'.format(epoch,num_iters), nrow=10)
