from torchvision import datasets , transforms
from torch.utils.data import DataLoader
import torch

class Cifar10Data:
    def __init__(self,):
        self.train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                    ])
        self.val_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                    ])
        self.trainset = datasets.CIFAR10(train=True,root = "data/",download=True,transform=self.train_transforms)
        self.valset  = datasets.CIFAR10(train=False , root="data/",download=True,transform=self.val_transforms)

        return None

    def dataloader(self,batch_size=128 ,num_workers = 4,device_count=torch.cuda.device_count()):
        loader_param = { "batch_size":batch_size,#*device_count,
                        "pin_memory":True,
                        "num_workers":num_workers,
                        "shuffle":True}

        trainLoader = DataLoader(self.trainset,**loader_param)
        valLoader = DataLoader(self.valset  ,**loader_param)
        return {"train":trainLoader , "val":valLoader}
    
    @property
    def num_classes(self,): return len(self.trainset.classes)
