from torchvision import transforms


def get_train_transforms(mean: tuple[float, float, float], std: tuple[float, float, float]):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_test_transforms(mean: tuple[float, float, float], std: tuple[float, float, float]):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
