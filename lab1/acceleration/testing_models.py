from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refactored_model_images"))

from utils import load_config, get_writer
from model import JeffreyEpsteinModel
from preprocessing import get_train_loader, get_test_loader
from train import evaluate, train
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleLogger")

batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

config = load_config()
for batch_size in batch_sizes:
    config["batch_size"] = batch_size
    model = JeffreyEpsteinModel().to(config["device"])
    train_loader = get_train_loader(config["batch_size"], size = 10000)
    test_loader = get_test_loader(config["batch_size"], size = 1000)
    writer = get_writer()
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(config["device"])
    train(model, config["epochs"], writer, train_loader, config["device"], config["lr"])
    evaluate(model, test_loader, config["device"])
    logger.info("Time taken for batch size %s: %.2f seconds | GPU mem: %.2f MB\n", batch_size, time.time() - start_time, torch.cuda.max_memory_allocated(config["device"]) / (1024 ** 2))
