from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refactored_model_images"))

from utils import load_config, get_writer
from model import JeffreyEpsteinModel
from preprocessing import get_train_loader, get_test_loader
from train import evaluate, train
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleLogger")

transformers = [
    ("No normalization", transforms.Compose([transforms.ToTensor()])),
    ("Normalize 0.5", transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])),
    ("Blur + Normalize 0.5", transforms.Compose([transforms.ToTensor(), transforms.GaussianBlur(kernel_size=3), transforms.Normalize((0.5,), (0.5,))])),
    ("Normalize 0.1", transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1,), (0.1,))]))
]

config = load_config()
for transform_name, transformer in transformers:
    model = JeffreyEpsteinModel().to(config["device"])
    train_loader = get_train_loader(config["batch_size"], transform=transformer, size = 1000)
    test_loader = get_test_loader(config["batch_size"], transform=transformer, size = 300)
    writer = get_writer()
    train(model, config["epochs"], writer, train_loader, config["device"], config["lr"])
    evaluate(model, test_loader, config["device"])
    logger.info(transform_name)
    print()
