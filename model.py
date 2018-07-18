import torch.nn as nn

 
class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(FrontEnd, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    output = self.main(x)
    return output


class D(nn.Module):

  def __init__(self,classes=10):
    super(D, self).__init__()
    
    self.main = nn.Sequential(
      nn.Conv2d(1024, 1, 1),
      nn.Sigmoid()
    )
    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    # self.conv_mu = nn.Conv2d(128, 2, 1)
    # self.conv_var = nn.Conv2d(128, 2, 1)

    self.cls = nn.Conv2d(1024,classes,1)

  def forward(self, x):
    output = self.main(x).view(-1, 1)
    logits = self.cls(x).view(-1,10)
    # y = self.conv(x)
    # mu = self.conv_mu(y).squeeze()
    # var = self.conv_var(y).squeeze().exp()

    return output, logits


class Q(nn.Module):

  def __init__(self,output_c=2):
    super(Q, self).__init__()

    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Conv2d(128, 10, 1)
    self.conv_mu = nn.Conv2d(128, output_c, 1)
    self.conv_var = nn.Conv2d(128, output_c, 1)

  def forward(self, x):

    y = self.conv(x)
    y = self.lReLU(self.bn(y))
    # disc_logits = self.conv_disc(y).squeeze()

    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    # return disc_logits, mu, var 
    return mu, var

class TotalD(nn.Module):
  
  def __init__(self):
    super(TotalD,self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True)
      )
    # REAL?
    self.real = nn.Sequential(
      nn.Conv2d(1024, 1, 1),
      nn.Sigmoid())

    self.FE = nn.Sequential(
      nn.Conv2d(1024, 128, 1, bias = False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1))

    self.disc = nn.Conv2d(128, 10, 1)
    self.mu   = nn.Conv2d(128, 2, 1)
    self.var  = nn.Conv2d(128, 2, 1)

  def forward(self,x):
    
    h = self.main(x)
    real = self.real(h).view(-1,1)
    
    # Q parameters
    fe = self.FE(h)
    disc_logits = self.disc(fe).view(-1,10)
    mu = self.mu(fe).squeeze()
    var = self.var(fe).squeeze().exp()

    return real, disc_logits, mu, var

class G(nn.Module):

  def __init__(self,input_=12):
    super(G, self).__init__()
    # Change Batch Norm to Instance Norm!

    self.main = nn.Sequential(
      nn.ConvTranspose2d(input_, 1024, 1, 1, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.main(x)

    
    return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)