from model import *
from trainer import Trainer
import torch

# TODO:
#  - Replace BatchNorm with Instance Norm
#  - L2 loss instead of factored Gaussian!?
#  - Reolicate DC GAN architecture

fe = FrontEnd()
d = D()
q = Q()
g = G()

for i in [fe, d, q, g]:
  if torch.cuda.is_available():
    i = i.cuda()
  i.apply(weights_init)

trainer = Trainer(g, fe, d, q)
trainer.train()
