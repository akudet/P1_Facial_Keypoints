## TODO: define the convolutional neural network architecture

import torch.nn as nn


# can use the below import should you choose to initialize the weights of your Net


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 224x224
        b1 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(1, 32, 5),
            # nn.Conv2d(1, 32, 3),
            # nn.Conv2d(32, 32, 3, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )
        # 56x56
        b2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(32, 32, 5, groups=32),
            # nn.Conv2d(32, 32, 3, groups=32),
            # nn.Conv2d(32, 32, 3, groups=32),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )
        # 14x14
        b3 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, 5, groups=64),
            # nn.Conv2d(64, 64, 3, groups=64),
            # nn.Conv2d(64, 64, 3, groups=64),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 7x7

        self.encoder = nn.Sequential(
            b1, b2, b3
        )

        self.decoder = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 68 * 2),
        )

        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))

        x = self.encoder(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.decoder(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
