import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
# from config import input_dim, output_dim

INPUT_DIM = 60
OUTPUT_DIM = 10

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out




