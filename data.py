from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from PIL import Image
import re

class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
        if self.training:
            self.train_path = path
            self.train_transforms = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(20),
                                    ])
            self.train_dataset = ImageFolder(root = self.train_path, transform = self.train_transforms)
        else:
            self.imgs = os.listdir(path)
            self.test_path = path
            self.test_transforms = transforms.Compose([
                                        transforms.Resize((224,224)),
                                   ])
    
    
    def __len__(self):
        return len(self.train_dataset)
        
    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        tensor_transforms = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
        
        if self.training:
            return (tensor_transforms(self.train_dataset[index][0]),self.train_dataset[index][1])
        else:
            image_loc = os.path.join(self.test_path, self.imgs[index])
            image = Image.open(image_loc).convert('RGB')
            tensor_image = tensor_transforms(self.test_transforms(image))
            return (tensor_image, )                              
