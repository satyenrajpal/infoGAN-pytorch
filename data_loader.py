import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.utils.data as data 
import os, glob
from PIL import Image
import torch

class RafD_dataset(data.Dataset):

    def __init__(self, image_dir, transforms=None):
        self.exprs = ['angry','comtemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.races = ['Caucasian', 'Moroccan']
        self.genders = ['female','male']
        self.image_dir = image_dir
        self.dataset = self.preprocess()
        self.transforms = transforms

    def preprocess(self):
        labels = []
        cwd = os.getcwd()

        os.chdir(self.image_dir)
        for i, filename in enumerate(glob.glob("*.jpg")):
            attrs = filename.split('_')

            if attrs[2] in self.races and arrs[3] in self.genders and attrs[4] in self.exprs:
                labels.append([filename,self.exprs.index(attrs[4]), self.genders.index(attrs[3]), self.races.index(attrs[2])])
        
        os.chdir(cwd)
        return labels

    def __getitem__(self, index):
        file, exp, gender, race = self.dataset[index]
        image = Image.open(os.join(self.image_dir, file))

        return self.transforms(image), torch.FloatTensor(exp), torch.FloatTensor([gender, race]) 
        
    def __len__(self):
        return len(self.dataset)


def get_loader(mode,image_size,batch_size,image_dir,num_workers,dataset,crop_size):
    
    transform_op=[]
    # Try without random horizontal flipping!!
    if mode == 'train' and dataset != 'MNIST':
        transform_op.append(T.RandomHorizontalFlip())
    
    if dataset != 'MNIST':
        transform_op.append(T.CenterCrop(crop_size))
    transform_op.append(T.Resize((image_size,image_size)))
    transform_op.append(T.ToTensor())
    transform_op.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
    # elif dataset=='MNIST':
    #     transform_op.append(T.Normalize(mean=[0.5], std=[0.5]))
    transform_op = T.Compose(transform_op)

    if dataset == 'MNIST':
        dataset = dset.MNIST(image_dir, transform=transform_op, download=True)
    elif dataset == 'RafD':
        dataset = ImageFolder(image_dir,transform_op) 

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=(mode=='train'),
                            num_workers=num_workers)
    return dataloader