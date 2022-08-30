## Network   Architecture
The network architecture is illustrated in Figure 1. It consists of a contracting 
path (left side) and an expansive path (right side). The contracting path follows 
the typical architecture of a convolutional network. It consists of the repeated 
application of two 3x3 convolutions (unpadded convolutions), each followed by 
a rectiﬁed linear unit (ReLU) and a 2x2 max pooling operation with stride 2 
for downsampling. At each downsampling step we double the number of feature 
channels. Every step in the expansive path consists of an upsampling of the 
feature map followed by a 2x2 convolution (“up-convolution”) that halves the 
number of feature channels, a concatenation with the correspondingly cropped 
feature map from the contracting path, and two 3x3 convolutions, each fol- 
lowed by a ReLU. The cropping is necessary due to the loss of border pixels in 
every convolution. At the ﬁnal layer a 1x1 convolution is used to map each 64- 
component feature vector to the desired number of classes. In total the network 
has 23 convolutional layers.
To allow a seamless tiling of the output segmentation map (see Figure 2), it 
is important to select the input tile size such that all 2x2 max-pooling operations 
are applied to a layer with an even x- and y-size.