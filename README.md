# CS 540 HW6 Spring 2023

This project is an extension on homework 6, this time consisting of the design and implementation of a convolutional neural network (the LeNet model) for the MiniPlaces dataset (scene recognition database with 120k images from 100 categories by MIT). This project references helper functions in `train_miniplaces.py` and `dataloader.py` and requires the packages listed in [homework 6](https://github.com/franklloydweft/CS540-HW6-Intro-PyTorch-Neural-Net) as well as some similar implementations.

## Packages used
- `from tqdm import tqdm`
- `import torch`
- `import torch.nn as nn`

## Implemented functions/classes
- `class LeNet(nn.Module)` including constructor outlining the LeNet layer architecture and `forward(self,x)` function given an input tensor `x` to transform through layers of LeNet and returning the shape of the transformed tensor in the form of a dictionary (`shape_dict`) and the tensor itself `x`
- `count_model_params()`: counts the number of trainable parameters and returns that number in the unit of millions.
- `train_model(model, train_loader, optimizer, criterion, epoch)`: uses similar implementation to HW6 to train model with parameters for the model, training data loader, an instance of an optimizer, loss function, and epoch number. Loops over standard training loop (HW6) and returns loss. This is the main "gimmick" of the assignment. Outputs of `test_model(model, test_loader, epoch)` are stored in the `results.txt` file in this repo showing the accuracy of the model after every training configuration.
- `test_model(model, test_loader, epoch)`: function for testing effects of training under different parameters, returning variable `test_acc` detailing accuracy of each for every epoch and test set.
  
### Training configurations (as detailed by the assignment)
This is the verbatim description of each configuration from the assignment document. Use it to reference `results.txt`.
> Please train the model under the following different configurations:
> 1. The default configuration provided in the code, which means you do not have to make modifications.
> 2. Set the batch size to 8, the remaining configurations are kept the same.
> 3. Set the batch size to 16, the remaining configurations are kept the same.
> 4. Set the learning rate to 0.05, the remaining configurations are kept the same.
> 5. Set the learning rate to 0.01, the remaining configurations are kept the same.
> 6. Set the epochs to 20, the remaining configurations are kept the same.
> 7. Set the epochs to 5, the remaining configurations are kept the same.
  
