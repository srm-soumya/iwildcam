from torchvision import datasets, transforms
from torch.utils.data import DataLoader

base_size = 256                         # Scale it to this size before cropping 
crop_size = 224                         # Input Size to the model
mean = [0.485, 0.456, 0.406]            # Imagenet mean
std = [0.229, 0.224, 0.225]             # Imagenet std


def create_transform():
    """Create training and validation transformation for the model."""
    transform = {
        'train': transforms.Compose([
            transforms.Resize(base_size),
            transforms.Pad(padding=2, padding_mode='reflect'),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(base_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    return transform


def create_dataloader(dir):
    """Create dataloader object for training and validation set.

    Args:
        dir: directory containing the train/ and valid/ folder
    Returns:
        tuple: (dataset, dataloader) 
    """
    transform = create_transform()
    dataset = {set: datasets.ImageFolder(
        root=dir/set, transform=transform[set]) for set in ['train', 'valid']}
    dataloader = {set: DataLoader(
        dataset[set], batch_size=60, shuffle=True) for set in ['train', 'valid']}
    return dataset, dataloader
