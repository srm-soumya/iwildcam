import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torchvision
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import models
from tqdm import tqdm

from model.loader import create_dataloader
from model.net import ResnetFinetuned


parser = argparse.ArgumentParser(
    'Specify the directory containing the dataset.')
parser.add_argument('--dir', '-d', required=True)
parser.add_argument('--num-epochs', '-n', default=3, type=int)
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str)
parser.add_argument('--dist-backend', default='gloo', type=str)
parser.add_argument('--rank', default=-1, type=int)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cwd = Path.cwd()


def train_model(model, dataset, dataloader, criterion, optimizer, scheduler, num_epochs):
    """Train the parameters of the model.

    Returns:
        Trained model
    """
    start = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}/{num_epochs - 1}')
        print('-'*10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            i = 0
            for inputs, labels in tqdm(dataloader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(mode=(phase == 'train')):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataset[phase])
            epoch_accuracy = (running_corrects.double() /
                              len(dataset[phase])) * 100

            print(
                f'Phase: {phase}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

            if phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    end = time.time()
    print(f'Training Time: {end - start}s')
    print(f'Best Validation Accuracy: {best_accuracy}')

    model.load_state_dict(best_model_weights)
    return model


def freeze(layers):
    """Freeze all the param in the model layers."""
    for param in layers.parameters():
        param.requires_grad = False


def save_model_weights(model, file):
    """Save the model weights.

    Args:
        model: pytorch model
        file: file name to store inside model/weights/
    """
    os.makedirs('model/weights', exist_ok=True)
    torch.save(model.state_dict(), cwd/'model'/'weights'/file)


def main(dir, args):
    # Create the loaders
    dataset, dataloader = create_dataloader(dir, args)

    # Load the pretrained resnet50 model
    resnet50 = models.resnet50(pretrained=True)
    # Create a custom head for the base resnet50 model
    model = ResnetFinetuned(model=resnet50)
    # Freeze the convolutional layers
    freeze(model.features)

    if not args.distributed:
        # If not distributed, run in parallel mode in the same machine
        model = nn.DataParallel(model).to(device)
    else:
        # If distributed run the model across different machines
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model)

    # Define the criterion, optimizer and scheduler
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.module.classifier.parameters(), lr=1e-2, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Train the model
    model = train_model(model, dataset, dataloader, criterion,
                        optimizer, scheduler, num_epochs=args.num_epochs)

    save_model_weights(model, 'resnet18-finetuned-weights.pth')


if __name__ == '__main__':
    args = parser.parse_args()
    args.distributed = args.world_size >= 2

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    dir = cwd/args.dir
    main(dir, args)
