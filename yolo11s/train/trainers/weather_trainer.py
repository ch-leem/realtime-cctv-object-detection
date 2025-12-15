# trainers/weather_trainer.py
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data import build_yolo_dataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import LOGGER
from datasets.weather_dataset import WeatherAugmentedDataset

class WeatherDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        dataset = build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=(mode == "val"),
            stride=gs,
        )

        if mode == "train":
            dataset = WeatherAugmentedDataset(dataset)
            LOGGER.info("Weather augmentation ENABLED")

        return dataset