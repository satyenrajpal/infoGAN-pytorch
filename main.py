from model import *
from trainer import Trainer
import torch

# TODO:
#  - Replace BatchNorm with Instance Norm
#  - L2 loss instead of factored Gaussian!?
#  - Reolicate DC GAN architecture
num_c = 4 
classes = 10
fe = FrontEnd()
d = D(classes = classes)
q = Q(output_c=num_c)
g = G(input_=classes+num_c)

for i in [fe, d, q, g]:
  if torch.cuda.is_available():
    i = i.cuda()
  i.apply(weights_init)

trainer = Trainer(g, fe, d, q,num_c=num_c)
trainer.train()
