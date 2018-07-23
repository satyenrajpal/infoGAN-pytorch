import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_loader(mode,image_size,batch_size,image_dir,num_workers,dataset,crop_size):
    
    transform_op=[]
    if mode == 'train' and dataset != 'MNIST':
        transform_op.append(T.RandomHorizontalFlip())
    if dataset != 'MNIST':
        transform_op.append(T.CenterCrop(crop_size))
    transform_op.append(T.Resize((image_size,image_size)))
    transform_op.append(T.ToTensor())
    if dataset != 'MNIST':
        transform_op.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
    transform_op = T.Compose(transform_op)

    if dataset == 'MNIST':
        dataset = dset.MNIST(image_dir, transform=transform_op, download=True)
    elif dataset == 'RafD':
        dataset = ImageFolder(image_dir,transform_op) 

    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=(mode=='train'),
                            num_workers=num_workers)
    return dataloader