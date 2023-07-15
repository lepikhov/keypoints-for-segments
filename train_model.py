import argparse
import sys

import config
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from dataset import prepare_samples
from model import HorseKeypointResNet50
from tqdm import tqdm

matplotlib.style.use('ggplot')


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--segment', help='must be in (head, neck, body, front leg, rear leg, horse)')
args = vars(parser.parse_args())

# segment
segment = args['segment']

match segment:
    case 'head': keypoints_number = len(config.HEAD_KEYPOINTS)
    case 'neck': keypoints_number = len(config.NECK_KEYPOINTS)
    case 'body': keypoints_number = len(config.BODY_KEYPOINTS)
    case 'front leg': keypoints_number = len(config.FRONTLEG_KEYPOINTS)
    case 'rear leg': keypoints_number = len(config.REARLEG_KEYPOINTS)
    case 'horse': keypoints_number = len(config.HORSE_KEYPOINTS)    
    case _:
        pass

# model
model = HorseKeypointResNet50(
    pretrained=True, requires_grad=True, keypoints_number=keypoints_number).to(config.DEVICE)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LR)
# we need a loss function which is good for regression like SmmothL1Loss ...
# ... or MSELoss
criterion = nn.SmoothL1Loss()

# training function


def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(
            config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss/counter
    return train_loss

# validatioon function


def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data['image'].to(
                config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if (epoch in [0, 2, 4, 6, 8]) or ((epoch+1) % 10 == 0):
                utils.valid_keypoints_plot(image, segment, outputs, keypoints, epoch)

    valid_loss = valid_running_loss/counter
    return valid_loss


if __name__ == "__main__":

    train_data, valid_data, train_loader, valid_loader = prepare_samples(
        segment)

    train_loss = []
    val_loss = []
    min_train_loss = sys.float_info.max
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1} of {config.EPOCHS}")
        train_epoch_loss = fit(model, train_loader, train_data)
        val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {val_epoch_loss:.4f}')

        if (epoch+1) % 10 == 0:
            # loss plots
            plt.figure(figsize=(10, 7))
            plt.plot(train_loss, color='orange', label='train loss')
            plt.plot(val_loss, color='red', label='validataion loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f"{config.ROOT_OUTPUT_DIRECTORY}/{segment}/loss.png")
            plt.show()

        if val_epoch_loss < min_train_loss:            

            min_train_loss = val_epoch_loss
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, f"{config.ROOT_OUTPUT_DIRECTORY}/{segment}/model.pth")
            
            
    print('DONE TRAINING')
