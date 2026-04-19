from utils import load_config, manage_config, get_writer
from model import JeffreyEpsteinModel
from preprocessing import get_train_loader, get_test_loader
from train import evaluate, train
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleLogger")

config = manage_config(load_config())
print("Final configuration:", config)
model = JeffreyEpsteinModel().to(config["device"])
train_loader = get_train_loader(config["batch_size"])
test_loader = get_test_loader(config["batch_size"])
writer = get_writer()
train(model, config["epochs"], writer, train_loader, config["device"], config["lr"])
evaluate(model, test_loader, config["device"])
