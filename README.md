# Deep-learning-in-matlab
This repo contains medium-hard deep learning projects done in matlab. This projects were done along the course 'FFR135 - Artificial neural networks' at chalmers. All 4 projects uses the CIFAR-10 dataset and can be loaded with the help of the matlab scripts LOADCIFAR and helperCIFAR10Data. Paste both of these scripts in the same folder as the project your running.

All projects contain a report which explain both the theoritcal results beside the experimental results acquired in the matlab code. 

## Fully connected networks
- Implements 4 netowrks of different complexity from **scratch** using standard data structures and operations in matlab e.g cell arrays, matrix multiplication etc..
- Analyzes the training progress for the different networks 

## Vanishing gradient 
- Implements a 'deep' network **from scratch** 
- Analyzes the learning speed for each layer
	- Deeper networks have an issue where the gradient is vanishing for the earlier layers causing a slow learning rate for earlier layers. This is more commonly known as vanishing gradient

## Regularization and early stopping
- Uses deep learning toolbox to implement the networks
- How does regularization affect training progress 

## Convolutional neural networks
- Uses deep learning toolbox to implement the networks
- How does depth and complexity affect performance 
	- Complexity as in the amount of *features maps* in each convolutional layer
