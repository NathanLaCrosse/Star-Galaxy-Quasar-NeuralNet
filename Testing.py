import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from Model import StarGalaxyData, StarGalaxyClassifier

# Load the dataset and neural network models
dat = StarGalaxyData()
clf_dict = torch.load("classifier1.pt", map_location=torch.device('cpu'))
sgq_clf = StarGalaxyClassifier()
sgq_clf.load_state_dict(clf_dict)

# Set the neural network to eval mode
sgq_clf.eval()

with torch.no_grad():
    # Print out accuracy data regarding the training, validation and test sets of data
    predictions = torch.argmax(sgq_clf(dat.train_features), dim=1)
    correct = (predictions == dat.train_labels).sum().item()
    print(f"Accuracy on train set: {correct / len(dat.train_features):.4f}")

    predictions = torch.argmax(sgq_clf(dat.validate_features), dim=1)
    correct = (predictions == dat.validate_labels).sum().item()
    print(f"Accuracy on validation set: {correct / len(dat.validate_features):.4f}")

    predictions = torch.argmax(sgq_clf(dat.test_features), dim=1)
    correct = (predictions == dat.test_labels).sum().item()
    print(f"Final Test Accuracy (Never Seen): {correct / len(dat.test_features)}")

    # Print out the confusion matrix of the model to understand what kind of decisions
    # it's making.
    cm = confusion_matrix(dat.test_labels, torch.argmax(sgq_clf(dat.test_features), dim=1).numpy(), normalize='true')
    disp_cm = ConfusionMatrixDisplay(cm, display_labels=['Star','Galaxy','Quasar'])
    disp_cm.plot()
    plt.show()