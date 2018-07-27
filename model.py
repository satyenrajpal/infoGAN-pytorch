import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class G(nn.Module):

  def __init__(self,input_=12,g_conv_dim=1024,image_size=32,output_c=3, res = False):
    super(G, self).__init__()

    layers = []
    layers.append(nn.ConvTranspose2d(input_, g_conv_dim, 1, 1, bias=False))
    # layers.append(nn.InstanceNorm2d(g_conv_dim))
    layers.append(nn.ReLU(True))
    
    curr_dim = g_conv_dim
    for i in range(int(math.log2(image_size))-1):
      layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1, bias=False))
      layers.append(nn.BatchNorm2d(curr_dim//2, affine=True,track_running_stats=True))
      layers.append(nn.ReLU(True))
      if res and i>3:
        layers.append(ResidualBlock(curr_dim//2, curr_dim//2))
      curr_dim = curr_dim//2

    layers.append(nn.ConvTranspose2d(curr_dim, output_c, 4, 2, 1, bias=False))
    layers.append(nn.Tanh())
    
    self.generate = nn.Sequential(*layers)
    self.image_size = image_size
  
  def forward(self, x):
    output = self.generate(x)
    assert output.size(2) == output.size(3) ==self.image_size
    return output


class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self, inp_c, conv_dim=64, image_size=32, lRelu_slope=0.1):
    super(FrontEnd, self).__init__()

    layers = []
    layers.append(nn.Conv2d(inp_c, conv_dim, 4, 2, 1))
    layers.append(nn.LeakyReLU(lRelu_slope, inplace=True)) #slope 0.01?
    
    curr_dim = conv_dim
    for i in range(int(math.log2(image_size)-1)):
      layers.append(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1))
      # layers.append(nn.BatchNorm2d(curr_dim*2))
      layers.append(nn.LeakyReLU(lRelu_slope, inplace=True)) #slope 0.01?
      curr_dim = curr_dim*2

    self.dim = curr_dim
    self.main = nn.Sequential(*layers)

  def forward(self, x):
    output = self.main(x)
    assert output.size(2) == output.size(3) == 1
    return output
  
  @property
  def end_dim(self):
    return self.dim
  

class D(nn.Module):

  def __init__(self, classes=10, FE_dim=1024):
    super(D, self).__init__()
    
    self.main = nn.Conv2d(FE_dim,   1,  1)
    self.cls  = nn.Conv2d(FE_dim,classes,1)
    self.classes = classes
  
  def forward(self, x):
    output = self.main(x).view(-1, 1)
    logits = self.cls(x).view(-1,self.classes)
    
    return output, logits

class Q(nn.Module):

  def __init__(self,output_c=2,FE_dim=1024):
    super(Q, self).__init__()

    self.conv = nn.Conv2d(FE_dim, 128, 1, bias=False)
    self.conv_mu = nn.Conv2d(128, output_c, 1)
    self.conv_var = nn.Conv2d(128, output_c, 1)

  def forward(self, x):

    y = self.conv(x)
    
    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()
 
    return mu, var


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)