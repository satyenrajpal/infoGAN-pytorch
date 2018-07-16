from model import *
from trainer import Trainer
import torch

fe = FrontEnd()
d = TotalD()
q = Q()
g = G()

for i in [fe, d, q, g]:
  if torch.cuda.is_available():
    i = i.cuda()
  i.apply(weights_init)

trainer = Trainer(g, fe, d, q)
trainer.train()
