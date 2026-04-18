import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse

class SUN397Dataset(Dataset):
    """
    A custom dataset class for loading the SUN397 dataset.
    """

    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset with images and labels.
        Args:
            data_dir (str): Path to the data directory.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir

        if not os.path.isdir(self.data_dir):
            raise ValueError(f"{data_dir} is not a valid directory")

        # Step 1: collect ALL class folders (second-level folders)
        self.class_dirs = []
        for top_level in os.listdir(self.data_dir):
            top_path = os.path.join(self.data_dir, top_level)

            if not os.path.isdir(top_path):
                continue

            for class_name in os.listdir(top_path):
                class_path = os.path.join(top_path, class_name)

                if os.path.isdir(class_path):
                    self.class_dirs.append((class_name, class_path))

        # Step 2: sort classes alphabetically by class name
        self.class_dirs = sorted(self.class_dirs, key=lambda x: x[0])

        # Step 3: create mapping
        self.class_to_idx = {
            class_name: idx for idx, (class_name, _) in enumerate(self.class_dirs)
        }

        # Step 4: collect all image paths + labels
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        self.samples = []
        for class_name, class_path in self.class_dirs:
            label = self.class_to_idx[class_name]

            for root, _, files in os.walk(class_path):
                for fname in files:
                    if os.path.splitext(fname)[1].lower() in valid_exts:
                        img_path = os.path.join(root, fname)
                        self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise ValueError("No images found in dataset")

        # Default transform
        if transform is None:
            mean, std = calculate_mean_std()
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the specified index.
        
        Args:
            idx (int): Index of the image to retrieve.
        
        Returns:
            tuple: (image, label)
        """
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
    


class CNN(nn.Module):
    """
    CNN with Global Average Pooling (improved architecture)
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.10),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.15),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.20),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
    
    
def calculate_mean_std(**kwargs):
    """
    Fill in the per channel mean and standard deviation of the dataset. 
    Just fill in the values, no need to compute them.
    """
    # return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]
    mean = [0.5127, 0.4529, 0.3974]
    std  = [0.2496, 0.2534, 0.2615]
    return mean, std

'''
All of the following functions are optional. They are provided to help you get started.
'''

def train(model, train_loader, val_loader=None, **kwargs):
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = kwargs.get("epochs", 40)
    lr = kwargs.get("lr", 0.025)
    save_path = kwargs.get("save_path", "model.pt")

    model.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = -1.0

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        if val_loader is not None:
            val_acc = test(model, val_loader, device=device)
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
        else:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}%"
            )

        scheduler.step()

    if val_loader is None:
        torch.save(model.state_dict(), save_path)
        
def test(model, test_loader, **kwargs):
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, 
                        default='welcome/to/CNN/homework',
                        help='Path to training data directory')
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
 
    return parser.parse_args()
    
def main():
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())
    else:
        print("Running on CPU")

    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean, std = calculate_mean_std()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 1. Dataset
    full_dataset = SUN397Dataset("./datanew", transform=None)

    # 2. Train/validation split
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )

    train_dataset_full = SUN397Dataset("./datanew", transform=train_transform)
    val_dataset_full = SUN397Dataset("./datanew", transform=val_transform)

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_subset.indices)

    # 3. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 4. Model
    model = CNN(num_classes=len(full_dataset.class_to_idx))

    # 5. Train and validate each epoch
    train(
        model,
        train_loader,
        val_loader=val_loader,
        device=device,
        epochs=40,
        lr=0.025,
        save_path="model.pt"
    )

    # 6. Final validation accuracy report
    model.load_state_dict(torch.load("model.pt", map_location=device))
    final_val_acc = test(model, val_loader, device=device)
    print(f"Best Model Validation Accuracy: {final_val_acc:.2f}%")


if __name__ == "__main__":
    main()