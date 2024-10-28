import os
import torch
import sys
import time
from torch import nn
import numpy as np


# Main Model Architecture
class TorchModel(nn.Module):
    def __init__(self, seq_len, num_lines, num_layers):
        super(TorchModel, self).__init__()
        # dataset dependencies:
        self.num_lines = num_lines
        self.seq_length = seq_len
        self.num_layers = num_layers

        # LSTM layer 1
        self.lstm_1 = nn.LSTM(input_size=num_lines, hidden_size=1000, batch_first=True)
        # LSTM layer 2
        self.lstm_2 = nn.LSTM(input_size=1000, hidden_size=500, batch_first=True)
        # Rest of the Neural Net
        self.fc_1 = nn.Linear(500, 3000)
        self.fc_2 = nn.Linear(3000, 1000)
        self.fc_3 = nn.Linear(1000, 3000)
        self.op_layer = nn.Linear(3000, 24 * num_lines)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm_1(x)  # first lstm layer
        output = self.sigmoid(output)
        # hn = hn.view(-1, 500)
        out, (hn, cn) = self.lstm_2(output)  # second lstm layer
        out = self.tanh(out)
        hn = hn.view(-1, 500)
        out = self.relu(hn)
        out = self.fc_1(out)  # first fc layer
        out = self.relu(out)
        out = self.fc_2(out)  # second fc layer
        out = self.relu(out)
        out = self.fc_3(out)  # third fc layer
        out = self.relu(out)
        out = self.op_layer(out)  # O/P layer
        out = self.sigmoid(out)

        return out
