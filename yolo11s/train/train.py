# train.py
from ultralytics import YOLO
from trainers.weather_trainer import WeatherDetectionTrainer
import torch
import yaml

# -------------------------
# Load config
# -------------------------
with open("configs/train_yolo11s.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
train_cfg = cfg["training"]
aug_cfg = cfg["augmentation"]

# -------------------------
# Load model
# -------------------------
model = YOLO(model_cfg["cfg"]).load(model_cfg["weights"])

args = dict(
    model=model_cfg["cfg"],
    data=cfg["data"]["path"],
    epochs=train_cfg["epochs"],
    imgsz=train_cfg["imgsz"],
    batch=train_cfg["batch"],
    device=train_cfg.get("device", "cuda"),
    save_dir=model_cfg["save_dir"],
)

# -------------------------
# Trainer 선택
# -------------------------
if aug_cfg.get("use_weather_aug", False):
    trainer = WeatherDetectionTrainer(overrides=args)
    trainer.train()
else:
    model.train(**args)
