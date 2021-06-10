import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from TrackDataset import TrackDataset
from converter import create_dataset, convert_table_to_track, split_dataset_dict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            #
            # nn.Linear(50, 100),
            # nn.Tanh(),

            nn.Linear(50, 250),
            nn.Tanh(),

            nn.Linear(250, 400),
            nn.Tanh(),

            nn.Linear(400, 4),
        )
        # self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     x = x.view(x.size(0), -1)
    #     y_hat = self.layers(x)
    #     loss = self.ce(y_hat, y)
    #     self.log('train_loss', loss)
    #     return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    #     return optimizer

