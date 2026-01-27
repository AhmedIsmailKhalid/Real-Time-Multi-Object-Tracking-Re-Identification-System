"""
Data augmentation and preprocessing transforms.
"""

import torchvision.transforms as T


def get_train_transforms(
    image_size: tuple[int, int] = (256, 128),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """
    Get training data transforms with augmentation.

    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean (R, G, B)
        std: Normalization std (R, G, B)

    Returns:
        Composed transforms
    """
    return T.Compose(
        [
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.5, scale=(0.02, 0.4), value="random"),
        ]
    )


def get_val_transforms(
    image_size: tuple[int, int] = (256, 128),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """
    Get validation/test data transforms (no augmentation).

    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean (R, G, B)
        std: Normalization std (R, G, B)

    Returns:
        Composed transforms
    """
    return T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
