import torch
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

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

class MLPDataSet(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]

        return x, y
    
    def __len__(self):
        return self.labels.shape[0]
    
train_dataset = MLPDataSet(X_train, y_train)
val_dataset = MLPDataSet(X_val, y_val)
test_dataset = MLPDataSet(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
