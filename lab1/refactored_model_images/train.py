import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import JeffreyEpsteinModel
import logging

logger = logging.getLogger("SimpleLogger")

def train(model : JeffreyEpsteinModel, epochs, writer, train_loader, device, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() # PyTorch accumulates gradients by default, so we need to clear them before computing new ones
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()   
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        if writer:
            writer.add_scalar("Loss/train", epoch_loss, epoch) #tb

        logger.info("Epoch %s, Loss: %.4f", epoch + 1, epoch_loss)

def evaluate(model : JeffreyEpsteinModel, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    logger.info("Accuracy: %.2f%%", 100 * correct / total)

