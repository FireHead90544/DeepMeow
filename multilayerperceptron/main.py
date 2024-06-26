import torch
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("xor_dataset.csv")
X = df[["x1", "x2"]].values
y = df["class label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train
)

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        
        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, 25), # 1st Hidden Layer
            torch.nn.ReLU(),

            torch.nn.Linear(25, 15), # 2nd Hidden Layer
            torch.nn.ReLU(),

            torch.nn.Linear(15, num_classes), # Output Layer
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits

    