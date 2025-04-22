import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Model import StarGalaxyData, StarGalaxyClassifier



""" --------------- Training Method --------------- """
def train_neural_classifier(epochs=10, batch_size=16, lr=0.001, trained_network=None, save_file="classifier1.pt"):
    # Set up data loader
    dat = StarGalaxyData()
    data_loader = DataLoader(dat, batch_size=batch_size, shuffle=True, drop_last=True)

    # Create the ANN and load save point from file if it exists
    sgq_clf = StarGalaxyClassifier()
    if trained_network is not None:
        sgq_clf.load_state_dict(torch.load(trained_network))
        sgq_clf.train()

    # Print the size of the neural network.
    print(f"Parameter Count: {sum(param.numel() for param in sgq_clf.parameters())}")

    # Initialize our loss function and the optimizer.
    cross_entropy = CrossEntropyLoss()
    optimizer = torch.optim.Adam(sgq_clf.parameters(), lr=lr)

    running_loss = 0.0

    # Run the epoch loop
    for epoch in range(epochs):
        for _, data in enumerate(tqdm(data_loader)):
            x, y = data

            # Run an optimizer step based off of the loss function
            optimizer.zero_grad()
            output = sgq_clf(x)
            loss = cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate how well the neural network is doing.
        with torch.no_grad():
            # Enter eval mode, which changes how batchnorm1d behaves.
            sgq_clf.eval()

            print(f"\nRunning loss for epoch {epoch + 1} of {epochs}: {running_loss:.4f}")

            # Measure accuracy on the training set
            predictions = torch.argmax(sgq_clf(dat.train_features), dim=1)
            correct = (predictions == dat.train_labels).sum().item()
            print(f"Accuracy on train set: {correct / len(dat.train_features):.4f}")

            # Measure accuracy on the validation set
            predictions = torch.argmax(sgq_clf(dat.validate_features), dim=1)
            correct = (predictions == dat.validate_labels).sum().item()
            print(f"Accuracy on validation set: {correct / len(dat.validate_features):.4f}")

        running_loss = 0.0

    torch.save(sgq_clf.state_dict(), save_file)
    return sgq_clf, dat

sgq_clf, dat = train_neural_classifier(epochs=1)

with torch.no_grad():
    predictions = torch.argmax(sgq_clf(dat.test_features), dim=1)
    correct = (predictions == dat.test_labels).sum().item()
    print(f"\n\nFinal Test Accuracy (Never Seen): {correct / len(dat.test_features)}")