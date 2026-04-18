import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from traian import SUN397Dataset, calculate_mean_std


def main():
    # Device print is optional, but helpful
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Use the same normalization as training
    mean, std = calculate_mean_std()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load dataset
    dataset = SUN397Dataset("./datanew", transform=transform)

    print(f"Total samples: {len(dataset)}")
    print(f"Total classes: {len(dataset.class_to_idx)}")
    print("Class to index mapping:")
    print(dataset.class_to_idx)

    # Check one sample
    image, label = dataset[0]
    print("\nSingle sample check:")
    print("Image type:", type(image))
    print("Label type:", type(label))
    print("Image shape:", image.shape)   # expected: [3, 224, 224]
    print("Label shape:", label.shape)   # expected: torch.Size([])
    print("Label value:", label.item())

    # Validation split to inspect behavior
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print("\nSplit check:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Check one batch from train loader
    train_images, train_labels = next(iter(train_loader))
    print("\nTrain batch check:")
    print("Train batch image shape:", train_images.shape)   # expected: [32, 3, 224, 224]
    print("Train batch label shape:", train_labels.shape)   # expected: [32]
    print("Train batch label dtype:", train_labels.dtype)

    # Check one batch from val loader
    val_images, val_labels = next(iter(val_loader))
    print("\nValidation batch check:")
    print("Val batch image shape:", val_images.shape)       # expected: [32, 3, 224, 224]
    print("Val batch label shape:", val_labels.shape)       # expected: [32]
    print("Val batch label dtype:", val_labels.dtype)

    # Sanity assertions
    assert image.shape == (3, 224, 224), f"Unexpected single image shape: {image.shape}"
    assert label.dtype == torch.long, f"Unexpected label dtype: {label.dtype}"
    assert train_images.shape[1:] == (3, 224, 224), f"Unexpected train batch image shape: {train_images.shape}"
    assert val_images.shape[1:] == (3, 224, 224), f"Unexpected val batch image shape: {val_images.shape}"
    assert train_labels.dtype == torch.long, f"Unexpected train label dtype: {train_labels.dtype}"
    assert val_labels.dtype == torch.long, f"Unexpected val label dtype: {val_labels.dtype}"

    print("\nDataset check passed successfully.")


if __name__ == "__main__":
    main()