import torch.nn as nn
import math
 
class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self, inp_c, conv_dim=128, image_size=32):
    super(FrontEnd, self).__init__()

    layers = []
    layers.append(nn.Conv2d(inp_c, conv_dim, 4, 2, 1))
    layers.append(nn.LeakyReLU(0.1, inplace=True)) #slope 0.01?
    
    curr_dim = conv_dim
    for i in range(int(math.log2(image_size)-1)):
      layers.append(nn.Conv2d(curr_dim, curr_dim//2, 4, 2, 1))
      layers.append(nn.BatchNorm2d(curr_dim//2))
      layers.append(nn.LeakyReLU(0.1, inplace=True)) #slope 0.01?
      curr_dim = curr_dim//2

    self.dim = curr_dim
    # self.main = nn.Sequential(
    #   nn.Conv2d(inp_c, 64, 4, 2, 1),
    #   nn.LeakyReLU(0.1, inplace=True),#16x16
    #   nn.Conv2d(64, 128, 4, 2, 1, bias=False),
    #   nn.BatchNorm2d(128), #8x8
    #   nn.LeakyReLU(0.1, inplace=True),
    #   nn.Conv2d(128, 256, 4, 2, 1, bias=False),
    #   nn.BatchNorm2d(256), #4x4
    #   nn.LeakyReLU(0.1, inplace=True),
    #   nn.Conv2d(256,1024,4,1,bias=False),
    #   nn.BatchNorm2d(1024), #1x1
    #   nn.LeakyReLU(0.1, inplace=True))
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
    # self.bn = nn.BatchNorm2d(128)
    # self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_mu = nn.Conv2d(128, output_c, 1)
    self.conv_var = nn.Conv2d(128, output_c, 1)

  def forward(self, x):

    y = self.conv(x)

    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()
 
    return mu, var

class G(nn.Module):

  def __init__(self,input_=12,g_conv_dim=1024,image_size=32,output_c=3):
    super(G, self).__init__()

    layers = []
    layers.append(nn.ConvTranspose2d(input_, g_conv_dim, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(g_conv_dim))
    layers.append(nn.ReLU(True))
    
    curr_dim = g_conv_dim
    for i in range(int(math.log2(image_size))-1):
      layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1, bias=False))
      layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True,track_running_stats=True))
      layers.append(nn.ReLU(True))
      curr_dim = curr_dim//2

    layers.append(nn.ConvTranspose2d(curr_dim, output_c, 4, 2, 1, bias=False))
    layers.append(nn.Sigmoid())
    
    self.generate = nn.Sequential(*layers)
    self.image_size = image_size
  
  def forward(self, x):
    output = self.generate(x)
    assert output.size(2) == output.size(3) ==self.image_size
    return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)