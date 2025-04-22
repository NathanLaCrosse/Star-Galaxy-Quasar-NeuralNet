# Star-Galaxy-Quasar-NeuralNet
Implements a neural network using pytorch to identify whether an object is a star, galaxy or quasar.

Uses data from the Sloan Digital Sky Survey, found here: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

This dataset consists of celestial objects and their light properties, which is used to classify a given source as either a star, galaxy or quasar. In this repository, I create a large neural network as a classifier to try to improve the results achieved in the SVC repository (link: https://github.com/NathanLaCrosse/Star-Galaxy-Quasar-SVC). One advantage a neural network has over the SVC is it was trained on the entire dataset in a reasonable time. The model was trained for 150 epochs on a NVIDIA gpu. Compared to the SVC results, the model achieves an additional 1% of accuracy, a minor yet substantial improvement. Examining the confusion matrix reveals that this model is better at predicting a quasar than the SVC, though the SVC performs better at predicting a galaxy, so there is a tradeoff present in this approach.
