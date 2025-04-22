import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

""" --------------- Dataset --------------- """
class StarGalaxyData(Dataset):
    def __init__(self):
        df = pd.read_csv("star_classification.csv")

        # This column causes troubles due to its name.
        df.rename(columns={'class': 'classification'}, inplace=True)

        # A lot of the columns have nothing to do with classifying the object.
        # For example, there is data of where the object is in the night sky and
        # different ids and dates for when it was recorded.
        df = df[["classification", "u", "g", "r", "i", "z", "redshift"]]

        # Get rid of rows with erroneous values.
        df = df.query("u >= 0")

        # Convert to a numpy array
        df = df.to_numpy()

        # Ensure the dataset is random but in a procedural way
        np.random.seed(0)
        np.random.shuffle(df)

        # Convert the string labels to numbers for easier handling.
        df[:,0][df[:, 0] == 'STAR'] = 0
        df[:, 0][df[:, 0] == 'GALAXY'] = 1
        df[:, 0][df[:, 0] == 'QSO'] = 2

        df = df.astype(np.float32)

        train_size = 80000
        validate_size = 10000
        self.len = train_size

        # Set up train, validate and test datasets
        self.train_features = torch.tensor(df[:train_size, 1:])
        self.validate_features = torch.tensor(df[train_size:train_size+validate_size, 1:])
        self.test_features = torch.tensor(df[train_size+validate_size:, 1:])

        self.train_labels = torch.tensor(df[:train_size, 0].astype(np.uint8))
        self.validate_labels = torch.tensor(df[train_size:train_size + validate_size, 0].astype(np.uint8))
        self.test_labels = torch.tensor(df[train_size+validate_size:, 0].astype(np.uint8))

    # Return an item in the training set when accessing the dataset.
    def __getitem__(self, item):
        return self.train_features[item], self.train_labels[item]

    def __len__(self):
        return self.len



""" --------------- Model --------------- """
class StarGalaxyClassifier(nn.Module):
    # Define the neural network's structure
    def __init__(self):
        super(StarGalaxyClassifier, self).__init__()

        # Some notes on the model structure.
        #  - Activation Functions - LeakyReLu performed the best in testing.
        #                         - Sigmoid has a disastrous performance.
        #                         - ReLu performs similar to LeakyReLu
        # - Size - Different amounts of nodes and hidden layers were tested.
        self.model = nn.Sequential(
            nn.BatchNorm1d(6),
            nn.Linear(6, 200),
            nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout1d(0.2),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout1d(0.2),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)