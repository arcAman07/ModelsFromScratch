## LeNet-5 Complete Architecture
LeNet-5, from the paper Gradient-Based Learning Applied to Document Recognition, is a very efficient convolutional neural network for handwritten character recognition.

Paper: Gradient-Based Learning Applied to Document Recognition

Authors: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
Published in: Proceedings of the IEEE (1998)

## Structure of the LeNet network
LeNet5 is a small network, it contains the basic modules of deep learning: convolutional layer, pooling layer, and full link layer. It is the basis of other deep learning models. Here we analyze LeNet5 in depth. At the same time, through example analysis, deepen the understanding of the convolutional layer and pooling layer.

![image](https://user-images.githubusercontent.com/76823502/187914722-c83bffd3-c55d-4e8a-801f-2de79660edef.png)

INPUT LAYER
The first is the data INPUT layer. The size of the input image is uniformly normalized to 32 * 32.

Note: This layer does not count as the network structure of LeNet-5. Traditionally, the input layer is not considered as one of the network hierarchy.

C1 layer-convolutional layer:
Input picture: 32 * 32
Convolution kernel size: 5 * 5
Convolution kernel types: 6
Output featuremap size: 28 * 28 (32-5 + 1) = 28
Number of neurons: 28 28 6
Trainable parameters: (5 5 + 1) 6 (5 * 5 = 25 unit parameters and one bias parameter per filter, a total of 6 filters)
Number of connections: (5 5 + 1) 6 28 28 = 122304
Detailed description:
The first convolution operation is performed on the input image (using 6 convolution kernels of size 5 5) to obtain 6 C1 feature maps (6 feature maps of size 28 28, 32-5 + 1 = 28).

Let's take a look at how many parameters are needed. The size of the convolution kernel is 5 5, and there are 6 (5 * 5 + 1) = 156 parameters in total, where +1 indicates that a kernel has a bias.

For the convolutional layer C1, each pixel in C1 is connected to 5 5 pixels and 1 bias in the input image, so there are 156 28 * 28 = 122304 connections in total. There are 122,304 connections, but we only need to learn 156 parameters, mainly through weight sharing.

S2 layer-pooling layer (downsampling layer):
Input: 28 * 28
Sampling area: 2 * 2
Sampling method: 4 inputs are added, multiplied by a trainable parameter, plus a trainable offset. Results via sigmoid
Sampling type: 6
Output featureMap size: 14 * 14 (28/2)
Number of neurons: 14 14 6
Trainable parameters: 2 * 6 (the weight of the sum + the offset)
Number of connections: (2 2 + 1) 6 14 14
The size of each feature map in S2 is 1/4 of the size of the feature map in C1.
Detailed description:
The pooling operation is followed immediately after the first convolution. Pooling is performed using 2 2 kernels, and S2, 6 feature maps of 14 14 (28/2 = 14) are obtained.

The pooling layer of S2 is the sum of the pixels in the 2 * 2 area in C1 multiplied by a weight coefficient plus an offset, and then the result is mapped again.

So each pooling core has two training parameters, so there are 2x6 = 12 training parameters, but there are 5x14x14x6 = 5880 connections.

C3 layer-convolutional layer:
Input: all 6 or several feature map combinations in S2
Convolution kernel size: 5 * 5
Convolution kernel type: 16
Output featureMap size: 10 * 10 (14-5 + 1) = 10
Each feature map in C3 is connected to all 6 or several feature maps in S2, indicating that the feature map of this layer is a different combination of the feature maps extracted from the previous layer.
One way is that the first 6 feature maps of C3 take 3 adjacent feature map subsets in S2 as input. The next 6 feature maps take 4 subsets of neighboring feature maps in S2 as input. The next three take the non-adjacent 4 feature map subsets as input. The last one takes all the feature maps in S2 as input.
The trainable parameters are: 6 (3 5 5 + 1) + 6 (4 5 5 + 1) + 3 (4 5 5 + 1) + 1 (6 5 5 +1) = 1516
Number of connections: 10 10 1516 = 151600
Detailed description:
After the first pooling, the second convolution, the output of the second convolution is C3, 16 10x10 feature maps, and the size of the convolution kernel is 5 5. We know that S2 has 6 14 14 feature maps, how to get 16 feature maps from 6 feature maps? Here are the 16 feature maps calculated by the special combination of the feature maps of S2. details as follows:

The first 6 feature maps of C3 (corresponding to the 6th column of the first red box in the figure above) are connected to the 3 feature maps connected to the S2 layer (the first red box in the above figure), and the next 6 feature maps are connected to the S2 layer The 4 feature maps are connected (the second red box in the figure above), the next 3 feature maps are connected with the 4 feature maps that are not connected at the S2 layer, and the last is connected with all the feature maps at the S2 layer. The convolution kernel size is still 5 5, so there are 6 (3 5 5 + 1) + 6 (4 5 5 + 1) + 3 (4 5 5 + 1) +1 (6 5 5 + 1) = 1516 parameters. The image size is 10 10, so there are 151600 connections.

