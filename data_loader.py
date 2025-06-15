# data_loader.py
from torchvision import datasets
from torch.utils.data import DataLoader
from data_preprocessing import train_transforms, val_transforms  # Import transformations

def load_data(train_pos_dir, train_neg_dir, val_pos_dir, val_neg_dir, batch_size=32):
    # Create a root directory for ImageFolder
    root_train = train_pos_dir.rsplit('\\', 1)[0]  # Get the parent directory of the positive class
    # Load training data using ImageFolder
    train_data = datasets.ImageFolder(root=root_train, transform=train_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Load validation data (similar logic)
    root_val = val_pos_dir.rsplit('\\', 1)[0]  # Get the parent directory for validation
    val_data = datasets.ImageFolder(root=root_val, transform=val_transforms)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

