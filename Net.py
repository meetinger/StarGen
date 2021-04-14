import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from converter import create_dataset, convert_table_to_track, split_dataset_dict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 500),
            # nn.ReLU(),
            nn.Linear(500, 500),
            # nn.ReLU(),
            nn.Linear(500, 500),
            # nn.ReLU(),
            nn.Linear(500, 500),
            # nn.ReLU(),
            nn.Linear(500, 4)
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


torch.manual_seed(42)

# track = convert_table_to_track('datasets/tracks/0001000M.track.eep')
track = convert_table_to_track('datasets/test.eep')

model = Net()

full_x, full_y = create_dataset(track, False)

print(full_x,)

# full_zip = dict(zip(full_x, full_y))
#
# # print(full_zip)
#
# train_data, test_data = split_dataset_dict(full_zip, 0.9)
#
# print(train_data, test_data)
#
# print(len(train_data), len(test_data))

# full_dataset = create_dataset(track)
#
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
#
# train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
#
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
# val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=1)
#
# model = Net()
#
# # loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#
# epoch = 30
# loss_func = torch.nn.MSELoss()
#
#
# model.eval()
# y_pred = model(val_dataset)
# before_train = criterion(y_pred.squeeze(), y_test)
# print('Test loss before training' , before_train.item())


# model.train()
# for epoch in range(epoch):
#     optimizer.zero_grad()  # Forward pass
#     y_pred = model(x_train)  # Compute Loss
#     loss = criterion(y_pred.squeeze(), y_train)
#
#     print('Epoch {}: train loss: {}'.format(epoch, loss.item()))  # Backward pass
#     loss.backward()
#     optimizer.step()
#
# # for i in range(epochs):
# #     for x, y in train_dataset:
# #         pass
#
# mean_train_losses = []
# mean_valid_losses = []
# valid_acc_list = []
