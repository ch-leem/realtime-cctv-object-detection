# datasets/weather_dataset.py
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from ultralytics.data.dataset import YOLODataset

class WeatherAugmentedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = dataset.labels  # YOLO 호환

        self.transform = A.Compose([
            A.RandomSnow(p=0.3),
            A.RandomFog(p=0.3),
            A.RandomRain(p=0.3),
            A.RandomShadow(p=0.3),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        img = batch["img"]

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()

        augmented = self.transform(image=img)
        batch["img"] = augmented["image"]

        return batch

    @staticmethod
    def collate_fn(batch):
        return YOLODataset.collate_fn(batch)