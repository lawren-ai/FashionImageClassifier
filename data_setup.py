

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor



def create_dataloaders(
        transform=transforms.ToTensor(),
        batch_size=32
):  
    # get training data
    train_data = datasets.FashionMNIST(root="data",
                                   train=True,
                                   download=True,
                                   transform=transform,
                                   target_transform=None
                                   )
    
    # get test data
    test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=transform,
                                  target_transform=None
                                  )
    
    class_names = train_data.classes

    # train dataloader
    train_dataloader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              )
    
    #test dataloader
    test_dataloader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)
    
    return train_dataloader, test_dataloader, class_names
    





    
